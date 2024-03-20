# coding: utf-8
import numpy as np


def smooth_curve(x):
    """?†?‹¤ ?•¨?ˆ˜?˜ ê·¸ë˜?”„ë¥? ë§¤ë„?Ÿ½ê²? ?•˜ê¸? ?œ„?•´ ?‚¬?š©
    
    ì°¸ê³ ï¼šhttp://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """?°?´?„°?…‹?„ ?’¤?„?Š”?‹¤.

    Parameters
    ----------
    x : ?›ˆ? ¨ ?°?´?„°
    t : ? •?‹µ ? ˆ?´ë¸?
    
    Returns
    -------
    x, t : ?’¤?„??? ?›ˆ? ¨ ?°?´?„°??? ? •?‹µ ? ˆ?´ë¸?
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """?‹¤?ˆ˜?˜ ?´ë¯¸ì??ë¥? ?…? ¥ë°›ì•„ 2ì°¨ì› ë°°ì—´ë¡? ë³??™˜?•œ?‹¤(?‰?ƒ„?™”).
    
    Parameters
    ----------
    input_data : 4ì°¨ì› ë°°ì—´ ?˜•?ƒœ?˜ ?…? ¥ ?°?´?„°(?´ë¯¸ì?? ?ˆ˜, ì±„ë„ ?ˆ˜, ?†’?´, ?„ˆë¹?)
    filter_h : ?•„?„°?˜ ?†’?´
    filter_w : ?•„?„°?˜ ?„ˆë¹?
    stride : ?Š¤?Š¸?¼?´?“œ
    pad : ?Œ¨?”©
    
    Returns
    -------
    col : 2ì°¨ì› ë°°ì—´
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2colê³? ë°˜ë??) 2ì°¨ì› ë°°ì—´?„ ?…? ¥ë°›ì•„ ?‹¤?ˆ˜?˜ ?´ë¯¸ì?? ë¬¶ìŒ?œ¼ë¡? ë³??™˜?•œ?‹¤.
    
    Parameters
    ----------
    col : 2ì°¨ì› ë°°ì—´(?…? ¥ ?°?´?„°)
    input_shape : ?›?˜ ?´ë¯¸ì?? ?°?´?„°?˜ ?˜•?ƒï¼ˆì˜ˆï¼?(10, 1, 28, 28)ï¼?
    filter_h : ?•„?„°?˜ ?†’?´
    filter_w : ?•„?„°?˜ ?„ˆë¹?
    stride : ?Š¤?Š¸?¼?´?“œ
    pad : ?Œ¨?”©
    
    Returns
    -------
    img : ë³??™˜?œ ?´ë¯¸ì???“¤
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
