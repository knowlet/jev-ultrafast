"""Publish path-redacted copies of local fixture evidence without changing inputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def requests(value):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'request':
                found.append(child)
            found.extend(requests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(requests(child))
    return found


def parse_documents(text, suffix):
    if suffix == '.json':
        return [json.loads(text)]
    if suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--dest-dir', type=Path, required=True)
    parser.add_argument('--path-map', type=Path, required=True,
                        help='Private JSON list of [absolute infrastructure prefix, public placeholder]')
    args = parser.parse_args()
    source = args.source_dir.resolve(strict=True)
    destination = args.dest_dir.resolve()
    if destination.exists() or destination.is_relative_to(source):
        parser.error('Destination must be fresh and outside the original results directory')
    mapping = json.loads(args.path_map.read_text())
    if not isinstance(mapping, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(item, str) for item in pair)
        or not pair[0].startswith('/') or not pair[1].startswith('<') for pair in mapping
    ):
        parser.error('Expected explicit infrastructure path-prefix mappings')
    mapping.sort(key=lambda pair: len(pair[0]), reverse=True)
    files, staged = [], []
    for original in sorted(source.rglob('*')):
        if not original.is_file() or original.suffix not in {'.json', '.jsonl', '.log'}:
            continue
        data = original.read_bytes()
        text = data.decode('utf-8')
        public, count = text, 0
        for private, replacement in mapping:
            count += public.count(private)
            public = public.replace(private, replacement)
        if re.search(r'/(?:Users|home)/[^/\s"<>]+', public):
            raise ValueError(f'Unmapped home path in {original.relative_to(source)}')
        if re.search(r'(?:Authorization\s*:\s*Bearer|sk-[A-Za-z0-9]{16,}|gh[opsu]_[A-Za-z0-9]{20,})', public):
            raise ValueError(f'Possible credential in {original.relative_to(source)}; no copy written')
        before, after = parse_documents(text, original.suffix), parse_documents(public, original.suffix)
        before_requests, after_requests = requests(before), requests(after)
        if before_requests != after_requests:
            raise ValueError(f'Redaction would change a model request in {original.relative_to(source)}')
        encoded = public.encode('utf-8')
        relative = original.relative_to(source)
        request_hashes = [digest(json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode())
                          for request in before_requests]
        files.append({'path': str(relative), 'original_sha256': digest(data), 'public_sha256': digest(encoded),
                      'original_bytes': len(data), 'public_bytes': len(encoded), 'path_replacements': count,
                      'model_requests_unchanged': True, 'request_sha256': request_hashes})
        staged.append((relative, encoded))
    destination.mkdir(parents=True)
    for relative, data in staged:
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    manifest = {'schema': 1, 'transformation': 'infrastructure-path-redaction-v1',
                'notes': ['Only known infrastructure path strings were replaced; original files remain untouched.',
                          'Numeric probabilities, timings, token counts and all model requests are unchanged.',
                          'Executed source hashes identify pre-publication local code; '
                          'portable paths and CI were added later.',
                          'Original hashes intentionally differ from redacted public copy hashes.'], 'files': files}
    (destination / 'sanitization-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Published {len(files)} sanitized evidence files; every model request unchanged')


if __name__ == '__main__':
    main()
