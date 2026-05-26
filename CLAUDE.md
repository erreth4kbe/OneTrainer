# Interactive DPO Loop System

## 프로젝트 개요

이 프로젝트는 **OneTrainer의 Diffusion-DPO (PR #1403) 위에 인터랙티브 모드를 추가**한다. 학습 엔진은 OneTrainer가 그대로 책임지고, OneTrainer 내부에 토글, 새 데이터 주입 경로, Pair Builder 팝업창을 끼워넣는다.

핵심 가치: **"출력 → 평가 → 학습 → 출력" 폐쇄 루프**가 사용자 입장에서 매끄럽게 도는 것. 시스템 자체의 인터랙티브함이 존재 이유.

## 핵심 설계 결정

### 데몬화하지 않는다 / 단일 프로세스

- OneTrainer의 학습 생명주기 그대로 유지. 데몬/IPC 없음.
- 페어 생성, 사용자 평가, 자동 등록, 학습 흡수를 **모두 OneTrainer 프로세스 안에서** 처리.
- 별도 FastAPI/SQLite/웹 UI 없음. customtkinter 팝업창(Pair Builder)이 사용자 인터페이스.

### Pair Builder가 모든 인터랙티브 입력의 중심

- RLHF 탭에서 Open Pair Builder 클릭 → 팝업 열림
- 팝업 안에서 prompt 입력, Generate Pair, Pick A/B, Start/Stop Training
- 메인 GUI Start 버튼은 PairBuilder 열려 있는 동안 lock (회색 disabled, 라벨/색상은 동기)

## 두 가지 시작 모드

### 기존 모드 (Interactive Mode OFF)

OneTrainer DPO 사용 방식 그대로. 사용자가 페어 데이터셋 사전 준비 → Start → 정해진 epoch → LoRA 저장 → 종료. PR #1403 그대로.

### 새 모드 (Interactive Mode ON)

- 페어 데이터셋이 시간에 따라 증가
- "1루프" = 사용자 정한 `epochs` 만큼. 매 루프 끝에 dataloader 통째 재생성 → 새 페어 자동 흡수
- `Total Loops`로 반복 횟수 지정 (-1이면 Stop까지 무한)
- 첫 루프 진입 시 step 시작 전 **wait 모드 (Ready)** — 사용자가 페어 만들고 Resume(Ready 버튼) 누름

## 베이스 학습 엔진: OneTrainer DPO PR (#1403)

- PR URL: https://github.com/Nerogar/OneTrainer/pull/1403
- 작업 브랜치: `pr-1403`

이 PR이 제공 (재구현 금지):
- Diffusion-DPO loss + reference model (frozen adapter snapshot)
- New Adapter / Existing Adapter 모드 자동 추론
- Sequential / Policy Concurrent / Full Concurrent 실행 모드
- DPO 전용 검증 메트릭, Early stopping
- 페어 데이터셋 폴더 구조 (`chosen/train/`, `chosen/val/`, `rejected/train/`, `rejected/val/`)
- `PairByFilename` 매칭 로직 (동일 stem)
- `BaseModelSetup.reference_model()` 컨텍스트 매니저 — 자기 증류에 재활용

## TrainConfig 새 필드

`modules/util/config/TrainConfig.py`:

| 필드 | 타입 | 디폴트 | 설명 |
|------|------|--------|------|
| `rlhf_interactive_mode` | bool | False | 인터랙티브 모드 ON/OFF |
| `rlhf_interactive_total_loops` | int | 1 | 총 루프 수. -1 = 무한. allow_negative 검증 |
| `rlhf_interactive_pairs_dir` | str | "" | Pair Builder가 페어를 dump할 폴더 경로 |
| `rlhf_interactive_cleanup_on_stop` | bool | False | 학습 종료 시 페어 폴더 + concept entry 정리 여부 |

Migration: `__migration_16`이 네 필드 setdefault.

## SampleConfig 새 필드

`modules/util/config/SampleConfig.py`:

| 필드 | 타입 | 디폴트 | 설명 |
|------|------|--------|------|
| `use_reference_model` | bool | False | sample 시 reference model 컨텍스트 사용 (자기 증류) |

## TrainCommands / TrainCallbacks 확장

- `TrainCommands.resume()` / `get_and_reset_resume_command()` — wait 모드에서 사용자 Resume 신호
- `TrainCallbacks.set_on_internal_state_changed(state)` — trainer가 "waiting"/"running" 상태 알림

## RLHF 탭 UI 추가

`modules/ui/RLHFTab.py`:
- Interactive Mode 토글 (`rlhf_interactive_mode`)
- Total Loops 입력 (`rlhf_interactive_total_loops`) — `-1` 입력 가능 (FieldValidator `allow_negative`)
- Interactive Pairs Folder 입력 + Browse (`rlhf_interactive_pairs_dir`)
- Open Pair Builder 버튼

## Pair Builder 팝업 (`modules/ui/DPOPairBuilderWindow.py`)

SampleWindow 베이스가 아닌 자체 구현. 신규 ~470줄.

### UI 구성
- Prompt + sample 설정 (SampleFrame 재활용, 2개)
- Action 버튼: `Generate Pair` / `Start Training` (메인과 양방향 동기)
- 이미지 A/B 슬롯 (`CTkFrame` container + `grid_propagate(False)` + `place(center)` → cell 측정 정확)
- Pick A / Pick B 버튼
- 진행 영역: progress bar + 상태 라벨 ("Generating A...") + 페어 카운터 ("Session: N   Total: M")
- Cleanup on Stop 토글

### 페어 생성 (`__generate_pair`)
- `commands.sample_custom(sample_a)` + `commands.sample_custom(sample_b)` — trainer가 step 사이에 처리
- `commands=None`(학습 안 됨)이면 "Start Training first" 안내
- **변별력 메커니즘**:
  - `cfg_scale`: base ± random(0.3, 1.0) (A는 +, B는 -)
  - seed: `random_seed=True`면 trainer가 매번 random, False면 sample_b만 `seed_a + random_offset`
  - **`sample_b.use_reference_model = True`** — B는 `reference_model` 컨텍스트 안에서 sample (NEW_ADAPTER 모드 = base, EXISTING_ADAPTER = frozen snapshot)
  - 결과: **A = current LoRA + cfg+delta**, **B = reference/base + cfg-delta**. 자기 증류 페어

### Pick (`__pick`)
- 자동 stem `pair_NNNNN` 명명
- chosen/rejected 폴더에 png + txt(prompt) dump
- 페어 카운터 갱신
- 자동 학습 시작 없음 — 사용자가 명시적으로 Ready/Start 클릭

### 이미지 표시 (`__display_image`)
- 매번 새 `CTkImage` 인스턴스 (customtkinter 캐싱 우회)
- `PIL.Image.resize` (확대/축소 양방향)
- 측정 기준: `container.winfo_width/height` (image 크기와 무관)

### Start/Stop 동기화
- PairBuilder의 train 버튼 클릭 → `train_ui.start_training()` 호출 (메인과 같은 함수)
- TrainUI의 state listener 메커니즘으로 양방향 반영
- 상태별 색상/라벨: idle=`Start Training`(녹색) / waiting=`Ready`(파랑) / running=`Stop`(빨강)

### lock_main_start_button
- PairBuilder 열림 → 메인 Start 버튼 disabled + 회색 (#6c757d). 라벨/색상 동기는 유지 (lock 시 fg_color override)
- 닫힘 → unlock + 현재 상태에 맞게 복귀

## ConceptType — 변경 없음

자동 등록은 기존 `DPO_CHOSEN`/`DPO_REJECTED` 타입 사용 + concept name `"RLHF Interactive Chosen/Rejected"`로 구별. 별도 enum 추가 안 함 (단순화).

## 자동 등록 / 정리

`dpo_curation_util.py`:
- **`ensure_interactive_concepts(concept_file, pairs_dir)`**: 학습 시작 시(`TrainUI.start_training`) 호출. `chosen/`, `rejected/` 폴더 생성 + concepts.json에 entry 두 개 자동 추가 (path+type 중복 검사)
- **`cleanup_interactive_concepts(concept_file, pairs_dir)`**: `cleanup_on_stop=True`일 때 학습 종료 후(`TrainUI.__training_thread_function`의 `trainer.end()` 다음) 호출. path+name 둘 다 일치하는 entry 제거 + chosen/rejected 폴더 `shutil.rmtree`

`ConfigList.reload_from_file()`: 파일 외부 변경 후 UI 강제 동기화. ensure 호출 후 `concepts_tab.reload_from_file()` 트리거.

## GenericTrainer 변경

`modules/trainer/GenericTrainer.py`:

### 외부 loop 루프
- `train()` 메서드 안 epoch loop를 `while True:` 외부 loop로 감쌈
- 매 루프 끝에 `self.data_loader = self.create_data_loader(...)` 재호출 (validation도)
- 두 번째 루프부터 `lr_scheduler = None`으로 reset
- 외부 loop 종료: 비인터랙티브면 1회 후 break, `total_loops=-1`이면 무한
- 진입 시 `[Interactive Loop N/M]` 콘솔 출력 (인터랙티브 모드에 한정)

### `__wait_for_user_resume()`
- 첫 외부 loop 진입 시(`loop_idx == 0` + 인터랙티브)만 호출
- `on_internal_state_changed("waiting")` → wait loop
- wait 안에서 `get_and_reset_sample_custom_commands` 처리 (페어 생성 가능)
- `get_and_reset_resume_command()` True면 종료 → `create_data_loader` 재빌드 → `on_internal_state_changed("running")` → 첫 epoch 시작

### sample 처리에 reference_model 분기
- `__sample_loop` 안 `_do_sample()` 헬퍼
- `sample_config.use_reference_model=True`면 `with self.model_setup.reference_model(self.model, self.config):` 컨텍스트 안에서 호출
- `RuntimeError`/`NotImplementedError` fallback: current 모델로 sample + 로그

## TrainUI 변경

`modules/ui/TrainUI.py`:
- `_TRAIN_BUTTON_STYLES`에 `"waiting"` 추가 (파랑, "Ready")
- `start_training()`의 `_is_waiting` 분기: pairs_dir/chosen/pair_*.png 카운트 → `< batch_size`면 `Cannot Resume` messagebox + return (drop_last로 0 step 방지)
- listener 메커니즘: `add/remove_training_state_listener`, `_broadcast_training_state`, `get_current_runtime`
- `_set_training_button_style`에 lock 가드 (fg_color/hover_color #6c757d 회색 + state disabled)
- `lock_main_start_button` / `unlock_main_start_button` 메서드
- `__on_trainer_internal_state`: "waiting"/"running" → `self.after(0, ...)` thread-safe dispatch
- `_main_start_button_locked` 초기화는 `bottom_bar` 호출 **전** (초기화 순서 보장)
- cleanup_on_stop 처리: trainer.end() 다음, `cleanup_interactive_concepts` + `concepts_tab.reload_from_file`

## FieldValidator 확장

`modules/util/ui/validation.py`:
- `FieldValidator(..., allow_negative=False)` — int/float 음수 검증을 옵션으로
- RLHFTab의 Total Loops entry가 `validator_factory`로 `allow_negative=True` 전달 → -1 입력 가능

## 동작 흐름

```
[idle] RLHF 탭에서 Interactive Mode + Pairs Folder 설정 → Open Pair Builder 클릭
   ↓
[PairBuilder 열림, 메인 Start 회색]
   PairBuilder Start 클릭 (페어 0개여도 가능 — PairByFilename이 인터랙티브 모드에서 빈 매칭 허용)
   ↓
[trainer.start] ensure_interactive_concepts (폴더+concept 자동 등록) → 모델 로드
   ↓
[trainer.train 진입] 외부 while 루프 → 첫 진입 wait 모드 진입
   ↓
[Ready 상태] (파랑 "Ready" 버튼)
   PairBuilder Generate Pair 클릭 → 2장 생성 (A=current LoRA + cfg+δ, B=reference + cfg-δ)
   사용자 Pick A/B → 페어 dump (chosen/rejected/pair_NNNNN)
   페어 누적 ≥ batch_size 되면 사용자가 Ready 클릭 → Resume
   ↓
[Resume] training_commands.resume() → wait 종료 → dataloader 재빌드 → 첫 step 시작
   ↓
[Stop 상태] (빨강 "Stop" 버튼) — 학습 진행 중
   사용자 추가 페어 생성 가능 (trainer.sample_custom으로 학습 중인 LoRA 적용 가시화)
   매 루프 끝에 dataloader 재생성 → 다음 루프 새 페어 흡수
   ↓
[Stop 클릭 또는 total_loops 도달] → trainer.end() → cleanup_on_stop=True면 정리
```

## 디렉토리 구조

```
project_root/               (OneTrainer 포크 = 이 디렉토리)
├── CLAUDE.md
├── modules/                (포크 소스)
└── workspace/              (OneTrainer 워크스페이스)
    ├── loras/
    ├── samples/
    └── {pairs_dir}/        (사용자가 RLHF 탭에서 지정)
        ├── chosen/
        │   └── pair_00001.{png,txt}
        └── rejected/
            └── pair_00001.{png,txt}
```

## 기술 스택

- Python 3.12 (OneTrainer venv)
- customtkinter
- OneTrainer 포크 (PR #1403 + 인터랙티브 패치)

## 결정된 사항

- OneTrainer 학습 루프 데몬화 X
- 외부 웹 시스템 X (단일 프로세스)
- DPO만 사용
- Lazy 베이스 모델 로드 모드 — **제거됨** (페어 0개도 학습 진입 가능하게 우회)
- 페어 생성: 동일 prompt, A=current LoRA / B=reference, cfg ± delta (자기 증류 + cfg 변별)
- 페어 등록: 자동 stem `pair_NNNNN`, 사용자 지정 폴더, concept name으로 인터랙티브임 명시
- 1루프 = `config.epochs`. 매 루프 끝 dataloader 통째 재생성
- Start 시 페어 0개 허용 (wait 모드에서 사용자가 페어 만들고 Resume)
- Resume 가드: pair < batch_size면 차단 (drop_last 0 step 방지)
- train만 자동 등록 (val은 사용자가 별도)
- Start/Stop 메인↔PairBuilder 양방향 동기 + PairBuilder 열림 시 메인 lock
- Latent caching은 사용자 설정 그대로

## 미결정 / 추후 과제

- 변별력 실제 효과 — 학습 일정 진행 후 reference vs current 차이가 사용자가 명확히 가릴 정도인지 (구현 완료, 효과는 장시간 테스트 필요)
- 학습 효과 측정 / drift 가시화 추가 옵션
- 페어 생성 옵션 추가 (sampler/step 차이 등)
- 디스크 정리 정책 (페어 누적, LoRA 버전)

## Phase 진행 상황

| Phase | 내용 | 상태 |
|---|---|---|
| 1-A | TrainConfig 2 필드 + RLHF 탭 토글 | ✅ |
| 1-B | GenericTrainer 외부 loop 루프 | ✅ |
| 1-C-1 | ensure_interactive_concepts + reload_from_file | ✅ |
| 1-C-2 | RLHF 탭 Pairs Folder 입력 | ✅ |
| 1-C-3 | DPOPairBuilderWindow 신규 + 진입 버튼 | ✅ |
| 1-C-4 | Start/Stop 양방향 동기화 | ✅ |
| 1-D-1 | TrainCommands.resume() + TrainConfig 추가 필드 | ✅ |
| 1-D-2 | GenericTrainer wait 모드 + TrainUI "waiting" 스타일 | ✅ |
| 1-D-3 | PairBuilder waiting 분기 + min_pairs/cleanup 위젯 | ✅ (min_pairs 후속 제거) |
| 1-D-4 | (lazy 모드 비동기) | ✅ (후속 제거) |
| 1-D-5 | cleanup_on_stop 처리 | ✅ |
| 후속 | lazy 모드 통째 제거 + batch_size 가드 | ✅ |
| 후속 | 메인 Start lock + 회색 처리 | ✅ |
| 후속 | 이미지 리사이즈 (container + grid_propagate) | ✅ |
| 후속 | progress bar + 상태 라벨 | ✅ |
| 후속 | -1 입력 가능 (FieldValidator allow_negative) + entry tooltip | ✅ |
| 후속 | 변별력: cfg ± delta + reference_model 자기 증류 | ✅ (테스트 필요) |
| Phase 2 | 30분+ 실제 사용 통합 테스트 | 진행 중 (사용자) |

## 환경

- OS: Windows 11
- GPU: RTX 5090
- 베이스 모델: Z-Image 추가학습본
- Python: OneTrainer venv

## 주의사항

- OneTrainer PR #1403 미머지 — master 머지 시 base 재설정 필요할 수도
- 학습 중 Generate Pair는 step 사이에 처리 — sample 빈도 ↑면 학습 속도 ↓
- 디스크 누적 (페어 이미지 + LoRA 버전들) 별도 정리 정책 필요
- `train_progress.epoch`은 외부 loop이라 무한 모드 시 매우 큰 값. 표시/저장 시 자릿수 가정 주의
- 매 루프 dataloader 재생성 시 latent caching 캐시는 디스크에 유지됨 (재사용)
