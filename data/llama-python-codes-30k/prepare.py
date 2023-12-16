from datasets import load_dataset


class LlamaPythonCodes30k(object):
    
    def __init__(self) -> None:
        self.dataset = self.download()
    
    def dataset_name(self):
        return "llama-python-codes-30k"
    
    def download(self):
        return load_dataset(self.dataset_name())
    
    
if __name__ == "__main__":
    dataset = LlamaPythonCodes30k()
    print(dataset.dataset)        

