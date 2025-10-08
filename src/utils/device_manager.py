import torch

class DeviceManager:
    @staticmethod
    def list_devices():
        """Prints and returns a list of available devices."""
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices = {num_devices}")

        devices = []
        if torch.cuda.is_available():
            devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            for i, dev in enumerate(devices):
                name = torch.cuda.get_device_name(i)
                print(f"Device {i} name: {name}\ndevices[{i}] = {dev}")
        else:
            devices = [torch.device("cpu")]
            print("No CUDA devices available \n")

        return devices
