from torch.utils.data import Dataset


class NuScenesDataSet(Dataset):

    def __init__(self, mode, split='training', img_list='trainval', is_training=True):

        self.mode = mode
        pass
