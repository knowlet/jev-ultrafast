# Q4 凍結 heldout 有界窗口：獨立核對（2026-10-13）

這次使用既有 System One Q4_K_M 真模型與隔離 Chromium 執行凍結測試，預定 24 次，只執行了第一輪前 6 次：2 成功、4 失敗、18 未執行。840 秒工作窗口到期後停止，含清理的總 wall 為 851.563946 秒（總上限 900 秒、清理預留 60 秒）；`complete=false`，runner exit 1。2/24＝8.33% 是整份預定表中已達成的成功比例，2/6＝33.33% 只描述已執行前綴，均不能當作完整泛化成功率。沒有 baseline arm，也沒有後續重跑或依結果調參。

本報告在執行窗口及原始記錄關閉後產生。獨立核對逐筆對照 request 物件、compact JSON SHA、回應及 local_bridge，並以凍結 checker 重算六個 fresh DOM；全部吻合原記錄，原始 raw 未改寫。完整 SHA、計數、未執行表及限制見 [audit](../benchmark/public/heldout-window-2026-10-13/audit/verified-summary.json)；執行證據見 [window receipt](../benchmark/public/heldout-window-2026-10-13/window-receipt.json)、[metadata](../benchmark/public/heldout-window-2026-10-13/heldout/metadata.json)、[runs](../benchmark/public/heldout-window-2026-10-13/heldout/runs.jsonl)、[summary](../benchmark/public/heldout-window-2026-10-13/heldout/summary.json)。

成功標準依原凍結規則：完成案例必須 genuine model DONE、fresh DOM 符合所有要求且無 execution error；負例必須 genuine model BLOCKED、零 browser actions、DOM 未改變且零 semantic interactions。checker 答案沒有送入模型。

| 第一次 repetition 0 | 結果 | logical / physical / client confirmed | browser actions | task / model / browser 秒 | 判定依據 |
|---|---|---:|---:|---:|---|
| repair_route | 失敗 | 4 / 25 / 24 | 3 | 179.413 / 179.262 / 0.148 | Save repair 已執行，但漏勾 Show checkpoints；180 秒任務期限到期。 |
| unknown_code | 成功 | 1 / 5 / 5 | 0 | 26.613 / 26.610 / 0.003 | genuine BLOCKED；空 code、零操作及 DOM 未變。 |
| complete_stamp | 成功 | 2 / 9 / 9 | 1 | 34.637 / 34.593 / 0.044 | genuine DONE；Violet stamp present、primary disabled、提交 1 次。 |
| form_bundle | 失敗 | 4 / 24 / 23 | 3 | 179.676 / 179.566 / 0.108 | DOM 全部要求已達成；尚無 genuine DONE，180 秒任務期限到期，仍為失敗。 |
| form_card | 失敗 | 2 / 13 / 12 | 1 | 131.041 / 131.005 / 0.035 | 只填 Juniper；第二次決策 SELECT target 在共用 90 秒期限取消。 |
| repair_preview | 失敗 | 3 / 17 / 16 | 2 | 177.229 / 177.138 / 0.090 | 填 Basil、選 Compact，但 checkbox/Apply 未完成；840 秒外部工作窗口中止。 |

未執行的是 repetition 0 的 `complete_latch`、`unsupported_document`，及全部 8 個任務的 repetition 1、2。原 `budget_exhausted=true` 共 3 次，分別是兩個 180 秒任務期限及一次外部工作窗口中止。`form_card` 原 error 為 `{"type":"NoulPolicyError","message":""}`、`budget_exhausted=false`，完整保留。

`form_card` 第二個 logical decision 為 90003.711500 ms；六個 operation HTTP 200/client confirmed 累計 80855.395875 ms，最後 SELECT target 等待 9133.073292 ms 後 CancelledError，整體 task 為 131041.320833 ms。分類依 [noul_policy.py](../jev_ultrafast/noul_policy.py) 的 choose_noul_async：所有 operation 與 target 共用一次 `deadline`，每次 HTTP 只使用剩餘時間；MAX_TIMEOUT=90。這不是依空白錯誤訊息猜測，也不是 180 秒 task timeout。

暖機有 2 次 development observation，每次 5 個 native requests 加 1 個 text helper，未混入測量統計。測量為 16 個 logical decisions、93 個 physical transports、89 個 client-confirmed native executions、4 個 helpers、10 個 browser actions。Wire 是 103 次 status ok＝10 warmup＋93 measured，99 次 client-confirmed＝10＋89。103 沒有超過事前 306 native 總上限；測量 93/288、暖機 10/18。

四個取消尾端皆能以原 request SHA 及順序精確配對 server 回應。這只證明 physical score 已完成；它們沒有 client HTTP response，不能補成成功 logical decision、genuine terminal 或 browser action。

| task | wire decision index（從 0） | phase / operation | client 等待 ms | server adapter latency ms | request SHA256 |
|---|---:|---|---:|---:|---|
| repair_route | 34 | operation / DONE | 4736.064 | 6890.092 | `cab65f3c6fd9de6e8b903344373b8bf5f669ee12e8b27435c9b0f075bbd69837` |
| form_bundle | 72 | operation / WAIT | 2103.465 | 6743.432 | `5207fc70aa4023de9e26dfac89aa2a2c4bc7fbb099596f23a2c76583eb6cc524` |
| form_card | 85 | target / SELECT | 9133.073 | 12922.250 | `59cd971c0220a693de0fd7cb3cb89c89c5eb563e8d983510f4d65938e986d5e8` |
| repair_preview | 102 | operation / WAIT | 5497.988 | 15870.718 | `a01a34ad00d054cd3a02e9791f4e2634bbfddaea1790548f2b7125de66c24668` |

時間以 ms 記錄，以下轉成秒顯示。所有 p95 使用 `(n-1)*q` 線性內插，只是小樣本描述值；包含 timeout/取消的異質樣本，不能宣稱穩定 p95 或較原失敗 baseline 快。model wall 計每個完整 logical policy/helper 一次，browser wall 計 task 中 outer fresh/observe/act 一次，不重複加入 native/CDP 子時間。

| 測量指標 | n | p50 秒 | p95 秒 |
|---|---:|---:|---:|
| 端到端 task（不含 setup/teardown） | 6 | 154.135 | 179.610 |
| 成功 task，含一個 BLOCKED、一個 DONE | 2 | 30.625 | 34.236 |
| 每個 logical decision，含未完成者 | 16 | 43.745 | 76.569 |
| 每個 physical client request，含取消 | 93 | 6.293 | 14.941 |
| 每 task model/helper wall | 6 | 154.071 | 179.490 |
| 每 task browser wall | 6 | 0.067 | 0.138 |
| streaming helper 首個 content delta | 4 | 3.458 | 5.757 |

非串流 offered-choice scoring 沒有 first-token event，decision TTFT 不可取得，且 autoregressive_tokens=0。4 個 helper 有真正 provider token counts；completion tokens / full request second 的 p50=3.242899、p95=4.841138，包含 prompt/transport。89 個 native client-confirmed requests 的 input tokens / adapter latency second p50=140.002756、p95=165.676426；它不是生成 tokens/s 或 pure GPU throughput。原凍結 adapter 沒有傳回獨立 native_latency_ms，此欄保持 null。

原模型為 `knowlet/Gemma-4-12B-Unified-System-One` 的 weights revision `66626de5cbcf8c5fecb8e9b58710805a02a42f67`，merged checkpoint SHA `fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d`；不是用 base Gemma 替代。GGUF `s1-boolq-Q4_K_M.gguf` 大小 7,381,383,584 bytes，SHA `daf56f5a5a0496e4e726e504858ef8dda1f6b02cd1bc7a68e9f1beb2bf178365`；native binary SHA `d1e81415da71954981ebc44794c90393961a8e150559a18afdf948d60f8506be`、原 adapter SHA `95e9343ead0656f13197c4371ab48b445826d48bd37f8cc9f03b94ae926351c1`。llama.cpp pin `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd`；ctx/batch/threads=4096/512/4，GPU layers=99、text seed=42、temperature=0、max_tokens=128；BoolQ T=3.232734824261939 沒有重新擬合，不能稱為導航 calibration，Noul p_true 也不是 categorical operation probability。prefix candidate 未使用。

Jev 記錄 HEAD `a59ac048c322b0fbe6734c5cd893e735825f1906`，工作分支 `feat/local-system-one-benchmark`；實際 source 以 SHA receipt 為準，不能只憑 HEAD 代表當時工作樹。plan SHA `e94986f9429889379b368c5adcbf4ab3fcfc00487879ee54be5133fb37f91196`、freeze SHA `ccbc97f8ecf5b29a3eac75bc5702695f72dcfbb2dc7ec33f217a1d34070c9041`；12 個凍結檔案獨立重算全符。執行 receipt 的 97 個 source/runtime/sidecar bookends 記錄不變；獨立輕量核對再讀 95 個不大於 1 MB 的來源檔，94 個吻合。窗口關閉後另有 cleanup 報告修正版 run_heldout_window.py，記錄的 SHA 為 `2622637a8dce6283183db92da910587550d0faa96266cccde0b9d069edf40655`，不同於本次執行版本 `e45fc7745dedb34e8b9137f6797d7f446d331f55e6c99e622c0524d3025dc988`，不把此新版本的行為倒填到本次結果。2 個較大 binary/tokenizer 沒有獨立重 hash。GGUF 使用 calibrated manifest 及原 adapter scoring 前驗證的 identity；原 controller 本身沒有 post-run 全 GGUF SHA。另有關閉後的 [post-package-integrity.json](../benchmark/public/heldout-window-2026-10-13/diagnostics/post-package-integrity.json)，CPU-only 讀取全部 9 個 manifest 檔案（含 7.38 GB GGUF、projector、tokenizer），5.613611 秒完成，所有 SHA 及 bytes 均吻合；獨立 audit 逐項對照該 receipt 與 manifest，沒有再次讀整份模型。原 [window controller](../benchmark/source-freezes/heldout-window-2026-10-13/scripts/run_heldout_window.py) 與 [heldout runner](../benchmark/source-freezes/heldout-window-2026-10-13/scripts/run_heldout_benchmark.py) 已在 [source freeze](../benchmark/source-freezes/heldout-window-2026-10-13/source-sha256.json) 保留，獨立重 hash 證明 byte exact。

硬體由本次 native Metal log 核實 Apple M1 Max，記憶體 bookend 均為 68,719,476,736 bytes（64 GiB）；macOS 27.0.1 arm64、Python 3.12.11、httpx 0.28.1、websockets 15.0.1。每次任務使用新的 owned Chromium profile，限制在該本地 fixture origin，沒有讀個人 cookies/帳號。

啟動 HTTP identity 為 262.077208 ms，但該 identity 是 starting，不能當成模型已完成 calibrated inference；text model readiness 4,855.200375 ms。第一個暖機 physical request client wall 22,236.806833 ms、adapter latency 11,918.847500 ms，含 process/startup/preparation，沒有控制 OS file cache，不是整台機器 cold-start 比較。

Native `/usr/bin/time -l` peak RSS 為 9,297,510,400 bytes，lifetime 833.97 秒；outer adapter 也是同樣 peak，lifetime 851.27 秒。outer 可能含 descendant rusage，不能稱 pure Python RSS，也不能把兩個 peak 相加。text helper direct PID wait4 peak 8,537,014,272 bytes，exit 0；runner direct wait4 peak 99,319,808 bytes、exit 1，亦可能含 descendants。benchmark 自己的 Python cumulative own-process peak 另為 53,772,288 bytes。六個 Chromium main PID 的 peak 91,095,040–99,319,808 bytes，exit 均 143（owned SIGTERM），不是自然 exit 0，也不是同時 process-tree RSS。

本窗口 memory pressure bookends free percentage 59%→75%；VM swapins 56→112、swapouts 25,700→25,700，所選累積計數沒有下降。這是全系統 snapshot，並行工作不受控，不把差值歸因給本測試。它沒有先前單次 development smoke 的 counter epoch discontinuity；也不因此推論 reboot 原因或整機 cold 可比較性。

清理驗證仍有限制。native 最後 raw event 為 `orderly_eof / exit_code=0`，text helper direct wait4 exit 0，支持各該模型程序關閉；六個 Chrome main PID 亦有 owned SIGTERM exit 143。但 `killpg(pgid,0)` 對 24238（decision-service）、24239（decision-adapter）、24241（text-service）的 liveness probe 遭 PermissionError，沒有重試。本次 [執行版 window controller](../benchmark/source-freezes/heldout-window-2026-10-13/scripts/run_heldout_window.py) SHA `e45fc7745dedb34e8b9137f6797d7f446d331f55e6c99e622c0524d3025dc988` 的 `alive_groups()`（當時第 235–247 行，關閉後已另有修改）會跳過 denied_groups，所以 receipt 的 `remaining_owned_groups=[]` 不證明全群組退出。完整群組存活狀態尚未確認；關閉後的核對沒有重試被拒操作、改走其他路徑探測或發送 signal。

公開證據 packet 包含 20 個檔案：17 份選取的 raw（11 份 results、6 份 diagnostics）、[source-selection.json](../benchmark/public/heldout-window-2026-10-13/source-selection.json)、[audit/verified-summary.json](../benchmark/public/heldout-window-2026-10-13/audit/verified-summary.json)，以及 [sanitization-manifest.json](../benchmark/public/heldout-window-2026-10-13/sanitization-manifest.json)。results 保留原相對檔名（含 `heldout/`）；diagnostic receipts 與 memory/VM snapshots 位於 `diagnostics/`。packet 不包含模型權重或瀏覽器 profile/cache。

公開副本由 [既有 sanitizer](../scripts/sanitize_local_results.py) 以 `infrastructure-path-redaction-v1` 處理，只替換已知本機 infrastructure path 字串。manifest 每筆列出 `original_sha256` → `public_sha256`、前後 bytes 及 path replacement 次數；原資料沒有覆寫。audit 裡的 `raw_files_sha256` 是原始檔案 SHA，不能拿來驗證已遮罩的公開檔案。sanitizer 會拒絕任何改變 model request 的遮罩，並記錄 `model_requests_unchanged=true` 與每個 compact JSON request SHA；因此本報告四個 request SHA 指的是未變的 request body，而非公開全檔 SHA。數字、機率、時序及模型請求仍是原測試資料。

離線驗證公開 packet 時，應讀取 manifest 並核對每個 `public_sha256`。以下同時重算 model request SHA，不啟動模型或程序探測。manifest 本身不列入自己的 hash ledger；其餘 19 份公開檔案都有條目。

```bash
export S1_PUBLIC_RAW="/path/to/public/heldout-window-packet"
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
packet = Path(os.environ["S1_PUBLIC_RAW"])
manifest = json.loads((packet / "sanitization-manifest.json").read_text())
assert manifest["schema"] == 1
assert manifest["transformation"] == "infrastructure-path-redaction-v1"
assert len(manifest["files"]) == 19

def requests(value):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "request":
                found.append(child)
            found.extend(requests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(requests(child))
    return found

for entry in manifest["files"]:
    path = packet / entry["path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["public_sha256"], entry["path"]
    assert len(data) == entry["public_bytes"], entry["path"]
    assert entry["model_requests_unchanged"] is True, entry["path"]
    text = data.decode("utf-8")
    documents = ([json.loads(text)] if path.suffix == ".json" else
                 [json.loads(line) for line in text.splitlines() if line.strip()]
                 if path.suffix == ".jsonl" else [])
    hashes = [hashlib.sha256(json.dumps(request, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
              for request in requests(documents)]
    assert hashes == entry["request_sha256"], entry["path"]

audit = json.loads((packet / "audit/verified-summary.json").read_text())
assert audit["counts"]["expected_runs"] == 24
assert audit["counts"]["native_server_completed_all"] == 103
assert audit["cleanup"]["full_group_liveness_verified"] is False
print("19 public SHA/size/request records match; 6/24 partial; group liveness unconfirmed")
PY
```

原有界啟動 recipe 使用可攜變數如下；此處只是記錄命令，本次核對沒有再執行。需保留本次 source pins，並在 repo 的 benchmark/results 下使用新的輸出目錄；若使用關閉後修正版 controller，cleanup 報告將使用關閉後修正，不能稱為本次 controller 的 byte-exact 重跑。這六個已揭露案例若用來調參，之後只能算 development；真正下一次泛化檢驗需另凍結未看過的案例。

```bash
export JEV_REPO="/path/to/jev-ultrafast"
export S1_REPO="/path/to/existing/System-One-repo"
export CHROME_PATH="/path/to/isolated/chrome-headless-shell"
export S1_WINDOW_OUTPUT="$JEV_REPO/benchmark/results/new-heldout-window"
"$JEV_REPO/.venv/bin/python" "$JEV_REPO/scripts/run_heldout_window.py" \
  --s1-repo "$S1_REPO" --chrome-path "$CHROME_PATH" \
  --output "$S1_WINDOW_OUTPUT" --execute
```

這次資料補上了真實多任務 browser 覆蓋，也顯示 serial Noul requests 在期限內無法穩定完成 form/repair。下一個因果步驟應在 development 資料獨立檢驗縮減 serial calls 或 deadline 處理的候選，保留 operation/target 語義與 DOM 驗收，再使用全新凍結案例；尚未完成的 prefix/cache 候選、訓練計畫及 baseline 比較均不算本次結果。舊 legacy timings、cache replay 與 software fixture checks 保持各自的歷史範圍。
