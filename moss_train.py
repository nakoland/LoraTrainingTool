# MOSS-Image LoRA 학습 실행 파일. moss_image_train_network.py 에 "중단 후 이어하기"만 더한다.
# 저장소 코드를 고치지 않으려고 트레이너를 상속한다. UI 의 학습 시작이 이 파일을 실행한다.
# - MOSS_STOP_FILE 이 가리키는 파일이 생기면 그 스텝이 끝난 뒤 상태 폴더를 저장하고 끝낸다.
# - --resume 으로 이어가면 sd-scripts 의 global_step 은 지금 에포크 안의 스텝부터 다시 세고 train_state.json 도 그 값으로 적는다.
#   그대로 두면 두 번째 이어하기부터 위치가 틀어지므로 실제 스텝을 따로 세어 다시 적는다.
import json
import os
import sys
import tomllib

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-scripts")
sys.path.insert(0, ROOT)
import moss_image_train_network as base
from library import checkpoint_io


class ResumableTrainer(base.MossImageNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.stop_file = os.environ.get("MOSS_STOP_FILE")
        self.prepared = False
        self.start_step = 0  # --resume 한 상태의 실제 스텝
        self.real_step = 0
        self.offset = None  # 실제 스텝 - global_step

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        if not self.prepared:
            self._prepare(accelerator, args)
        super().sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)
        if epoch is not None:  # 스텝이 끝날 때는 None, 학습 전과 에포크 끝에는 번호로 불린다
            return
        if self.offset is None:
            self.offset = self.start_step + 1 - global_step
        self.real_step = global_step + self.offset
        if self.stop_file and os.path.exists(self.stop_file):
            self._stop(accelerator, args)

    def _prepare(self, accelerator, args):
        self.prepared = True
        if args.resume:
            with open(os.path.join(args.resume, "train_state.json"), encoding="utf-8") as f:
                self.start_step = self.real_step = json.load(f)["current_step"]
            print(f"moss_resume step={self.start_step} total={args.max_train_steps}", flush=True)
        save_state = accelerator.save_state

        def save_and_note(output_dir=None, *a, **k):
            result = save_state(output_dir, *a, **k)
            self._note_state(output_dir, args)
            return result

        accelerator.save_state = save_and_note

    def _note_state(self, state_dir, args):
        """train_state.json 에 실제 스텝과, 이어할 때 비교할 설정을 적는다."""
        path = os.path.join(state_dir, "train_state.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        with open(args.dataset_config, "rb") as f:
            ds = tomllib.load(f)
        batch = ds.get("datasets", [{}])[0].get("batch_size", ds.get("general", {}).get("batch_size", 1))
        data.update(current_step=self.real_step, total_steps=args.max_train_steps,
                    settings={"network_dim": args.network_dim, "network_alpha": args.network_alpha, "batch_size": batch})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _stop(self, accelerator, args):
        name = checkpoint_io.STEP_STATE_NAME.format(args.output_name or checkpoint_io.DEFAULT_STEP_NAME, self.real_step)
        state_dir = os.path.join(args.output_dir, name)
        accelerator.save_state(state_dir)
        accelerator.end_training()
        print(f"moss_stopped {state_dir}", flush=True)
        raise SystemExit(0)


if __name__ == "__main__":
    parser = base.setup_parser()
    args = parser.parse_args()
    base.args_util.verify_command_line_training_args(args)
    args = base.args_util.read_config_from_file(args, parser)
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    if args.attn_mode is None and not args.xformers:
        args.attn_mode = "flash"  # 설정 파일에 없으면 FA2 다. UI 스위치의 기본값과 같다
    if args.show_timesteps:
        base.moss_image_train_utils.show_timesteps(args)
    else:
        ResumableTrainer().train(args)
