import torch


def main() -> None:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cudnn_available={torch.backends.cudnn.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_version={torch.version.cuda}")
        print(f"cudnn_version={torch.backends.cudnn.version()}")
        print(f"device_count={torch.cuda.device_count()}")
        print(f"device_name={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
