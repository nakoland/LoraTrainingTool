"""Add epoch-based caption selection to the pinned sd-scripts installation."""
import os
from pathlib import Path


def with_epoch_caption_selection(source):
    marker = "candidate_index = _select_moss_caption(len(cached_captions), getattr(self, 'caption_candidate_epoch', None))"
    if "def _select_moss_caption(" in source and marker in source:
        return source
    helper = """def _select_moss_caption(count, epoch):
    if count <= 1:
        return 0
    if epoch is None:
        return random.randrange(count)
    # The worker selects a caption before the collator sees the batch.
    return (max(1, epoch.value) - 1) % count


"""
    definition = "class BaseDataset(torch.utils.data.Dataset):"
    edits = (
        (definition, helper + definition),
        ("caption = random.choice(candidates) if len(candidates) > 1 else candidates[0]",
         "caption = candidates[_select_moss_caption(len(candidates), getattr(self, 'caption_candidate_epoch', None))]"),
        ("candidate_index = random.randrange(len(cached_captions)) if len(cached_captions) > 1 else 0", marker),
        ("one of them is chosen randomly at each step", "one of them is selected for each training sample"),
        ("ステップごとにランダムに選びます", "学習サンプルごとに候補を選びます"),
    )
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("Incompatible sd-scripts/library/dataset.py. The installed file was left intact.")
        source = source.replace(before, after, 1)
    return source


def main():
    path = Path(__file__).resolve().parents[1] / "sd-scripts" / "library" / "dataset.py"
    source = path.read_text(encoding="utf-8")
    updated = with_epoch_caption_selection(source)
    if source == updated:
        return
    compile(updated, str(path), "exec")
    temporary = path.with_suffix(".moss-patch.tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
