"""Write a Kaggle submission from a saved checkpoint (EMA weights + flip-TTA, same as the end of train.py).

Usage: python predict.py best_t2_400ep.pt submission_t2_400ep.csv
"""
import csv, sys
import torch
import torchvision.transforms as T
from train import ResNet, PNGs, read_rows, predict, n_params, DATA, from_ckpt

if __name__ == "__main__":
    ckpt, out = sys.argv[1], sys.argv[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev)
    if "sd" in ck:
        model = from_ckpt(ck)
        model.load_state_dict(ck["sd"])
    else:  # run 1 (best.pt) is a bare state dict of the default architecture
        model = ResNet()
        model.load_state_dict(ck)
    model = model.to(dev).eval()
    assert n_params(model) <= 5_000_000, "over the 5M budget"

    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    test = PNGs(read_rows(DATA / "test.csv", "id"), DATA / "test_images", tf)
    with torch.no_grad(), open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for x, ids in torch.utils.data.DataLoader(test, 500):
            for i, p in zip(ids.tolist(), predict(model, x.to(dev)).argmax(1).tolist()):
                w.writerow([i, p])
    print(f"wrote {out} from {ckpt} ({n_params(model):,} params)")
