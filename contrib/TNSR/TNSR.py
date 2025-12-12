import torch

class TSNR:
    @staticmethod
    def unfold(tensor, mode):
        return tensor.permute(*([mode] + list(range(mode)) + list(range(mode + 1, tensor.dim())))).view(tensor.size(mode), -1)
    
    @staticmethod
    def fold(tensor, mode, shape):
        full_shape = list(shape)
        mode_dim = full_shape.pop(mode)
        full_shape.insert(0, mode_dim)

        if None in full_shape:
            full_shape[full_shape.index(None)] = -1

        return tensor.view(*full_shape).permute(*list(range(mode + 1, tensor.dim())) + [mode] + list(range(mode)))

    @staticmethod
    def mode_dot(tensor, matrix, mode):
        new_shape = list(tensor.shape)

        if matrix.shape[1]!= tensor.shape[mode]:
            raise ValueError("Shape error. {0}(matrix's 2nd dimension) is not as same as {1} (dimension of the tensor)".format(matrix.shape[1], tensor.shape[mode]))

        new_shape[mode] = matrix.shape[0]

        res = torch.matmul(matrix, TNSR.unfold(tensor, mode))

        return TNSR.fold(res, mode, new_shape)

    @staticmethod
    def tucker_to_tensor(core, factors):
        for i, factor in enumerate(factors):
            core = TNSR.mode_dot(core, factor, i)
        return core

    @staticmethod
    def tt_to_tensor(cores):
        tensor_size = [c.shape[1] for c in cores]

        md = 2
        new_shape = cores[0].shape[:-1] + cores[1].shape[1:]
        t = TNSR.mode_dot(cores[0], cores[1].transpose(0, 1), md).view(new_shape)

        for i in range(1, len(tensor_size) - 1):
            md += 1
            new_shape = t.shape[:-1] + cores[i + 1].shape[1:]

            t = TNSR.mode_dot(t, cores[i + 1].transpose(0, 1), md).view(new_shape)

        return t.view(*tensor_size)

    @staticmethod
    def cp_to_tensor(rank1_tnsrs):
        tnsr = rank1_tnsrs[0][0]
        for i in range(1, len(rank1_tnsrs[0])):
            tnsr = torch.matmul(torch.reshape(tnsr, (-1, 1)), torch.reshape(rank1_tnsrs[0][i], (1, -1)))

        for j in range(1, len(rank1_tnsrs)):
            t = rank1_tnsrs[j][0]
            for k in range(1, len(rank1_tnsrs[j])):
                t = torch.matmul(torch.reshape(t, (-1, 1)), torch.reshape(rank1_tnsrs[j][k], (1, -1)))
            tnsr = torch.add(tnsr, t)

        return tnsr

    @staticmethod
    def np_unfold(tensor, mode):
        return tensor.permute(*([mode] + list(range(mode)) + list(range(mode + 1, tensor.dim())))).view(tensor.size(mode), -1)

    @staticmethod
    def np_fold(unfolded_tensor, mode, shape):
        full_shape = list(shape)
        mode_dim = full_shape.pop(mode)
        full_shape.insert(0, mode_dim)

        if None in full_shape:
            full_shape[full_shape.index(None)] = -1

        return unfolded_tensor.view(*full_shape).permute(*list(range(mode + 1, unfolded_tensor.dim())) + [mode] + list(range(mode)))

    @staticmethod
    def np_mode_dot(tensor, matrix, mode):
        new_shape = list(tensor.shape)

        if matrix.shape[1]!= tensor.shape[mode]:
            raise ValueError("Shape error. {0}(matrix's 2nd dimension) is not as same as {1} (dimension of the tensor)".format(matrix.shape[1], tensor.shape[mode]))

        new_shape[mode] = matrix.shape[0]

        res = torch.matmul(matrix, TNSR.np_unfold(tensor, mode))

        return TNSR.np_fold(res, mode, new_shape)