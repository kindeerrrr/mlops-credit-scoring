import torch
import numpy as np

def main():
    model = torch.load("model/model.pt", map_location="cpu")
    model.eval()

    x = torch.randn(1, 10)
    with torch.no_grad():
        y = model(x)

    print("Inference OK, output:", y.numpy())

if __name__ == "__main__":
    main()
