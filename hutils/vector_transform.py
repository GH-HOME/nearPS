import numpy as np


def normalizeVectorSet(vectorset):
    """
    normalize the vector set
    vectorset: with shape [N, dim]
    """
    assert vectorset is not None
    return vectorset / np.linalg.norm(vectorset, axis = 1, keepdims=True)

def extract_common_elements_fast(source, target):
    """
    Find the common elements in source and target lists and return the index of element in source and target array
    :param source: [N, 2] list with N elements and each element has dim k
    :param target: [N, 2]
    :return: [s, k] the common elements [s, 2] for common index for source and target list seperately
    """
    assert source is not None
    assert target is not None

    source_view = source.view([('',source.dtype)]*source.shape[1])
    target_view = target.view([('',target.dtype)]*target.shape[1])
    result = np.intersect1d(source_view, target_view, assume_unique=True, return_indices=True)
    common_eles = result[0].view(source.dtype).reshape(-1, source.shape[1])
    src_id = result[1].astype(np.int)
    tar_id = result[2].astype(np.int)
    index = np.dstack([src_id, tar_id]).squeeze()
    return [common_eles, index]


def extract_common_elements_fast_1d(source, target):
    """
    Find the common elements in source and target lists and return the index of element in source and target array
    :param source: [N, 2] list with N elements and each element has dim k
    :param target: [N, 2]
    :return: [s, k] the common elements [s, 2] for common index for source and target list seperately
    """
    assert source is not None
    assert target is not None

    result = np.intersect1d(source, target, assume_unique=True, return_indices=True)
    common_eles = result[0]
    src_id = result[1].astype(np.int)
    tar_id = result[2].astype(np.int)
    index = np.dstack([src_id, tar_id]).squeeze()
    return [common_eles, index]


def buildCrossMatrix(vec):
    """
    build matrix for cross product
    vec vector with dim 3
    return [3, 3] matrix
    """
    assert len(vec) == 3
    [a1, a2, a3] = vec
    matrix = np.zeros([3, 3], dtype= np.float32)
    matrix[0, 1] = -a3
    matrix[0, 2] = a2
    matrix[1, 0] = a3
    matrix[1, 2] = -a1
    matrix[2, 0] = -a2
    matrix[2, 1] = a1
    return matrix



