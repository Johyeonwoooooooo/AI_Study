# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # ë¶?ëª? ?””? ‰?„°ë¦¬ì˜ ?ŒŒ?¼?„ ê°?? ¸?˜¬ ?ˆ˜ ?ˆ?„ë¡? ?„¤? •
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """?™„? „?—°ê²? ?‹¤ì¸? ?‹ ê²½ë§

    Parameters
    ----------
    input_size : ?…? ¥ ?¬ê¸°ï¼ˆMNIST?˜ ê²½ìš°?—” 784ï¼?
    hidden_size_list : ê°? ????‹‰ì¸µì˜ ?‰´?Ÿ° ?ˆ˜ë¥? ?‹´??? ë¦¬ìŠ¤?Š¸ï¼ˆe.g. [100, 100, 100]ï¼?
    output_size : ì¶œë ¥ ?¬ê¸°ï¼ˆMNIST?˜ ê²½ìš°?—” 10ï¼?
    activation : ?™œ?„±?™” ?•¨?ˆ˜ - 'relu' ?˜¹??? 'sigmoid'
    weight_init_std : ê°?ì¤‘ì¹˜?˜ ?‘œì¤??¸ì°? ì§?? •ï¼ˆe.g. 0.01ï¼?
        'relu'?‚˜ 'he'ë¡? ì§?? •?•˜ë©? 'He ì´ˆê¹ƒê°?'?œ¼ë¡? ?„¤? •
        'sigmoid'?‚˜ 'xavier'ë¡? ì§?? •?•˜ë©? 'Xavier ì´ˆê¹ƒê°?'?œ¼ë¡? ?„¤? •
    weight_decay_lambda : ê°?ì¤‘ì¹˜ ê°ì†Œ(L2 ë²•ì¹™)?˜ ?„¸ê¸?
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # ê°?ì¤‘ì¹˜ ì´ˆê¸°?™”
        self.__init_weight(weight_init_std)

        # ê³„ì¸µ ?ƒ?„±
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """ê°?ì¤‘ì¹˜ ì´ˆê¸°?™”
        
        Parameters
        ----------
        weight_init_std : ê°?ì¤‘ì¹˜?˜ ?‘œì¤??¸ì°? ì§?? •ï¼ˆe.g. 0.01ï¼?
            'relu'?‚˜ 'he'ë¡? ì§?? •?•˜ë©? 'He ì´ˆê¹ƒê°?'?œ¼ë¡? ?„¤? •
            'sigmoid'?‚˜ 'xavier'ë¡? ì§?? •?•˜ë©? 'Xavier ì´ˆê¹ƒê°?'?œ¼ë¡? ?„¤? •
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUë¥? ?‚¬?š©?•  ?•Œ?˜ ê¶Œì¥ ì´ˆê¹ƒê°?
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidë¥? ?‚¬?š©?•  ?•Œ?˜ ê¶Œì¥ ì´ˆê¹ƒê°?
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """?†?‹¤ ?•¨?ˆ˜ë¥? êµ¬í•œ?‹¤.
        
        Parameters
        ----------
        x : ?…? ¥ ?°?´?„°
        t : ? •?‹µ ? ˆ?´ë¸? 
        
        Returns
        -------
        ?†?‹¤ ?•¨?ˆ˜?˜ ê°?
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """ê¸°ìš¸ê¸°ë?? êµ¬í•œ?‹¤(?ˆ˜ì¹? ë¯¸ë¶„).
        
        Parameters
        ----------
        x : ?…? ¥ ?°?´?„°
        t : ? •?‹µ ? ˆ?´ë¸?
        
        Returns
        -------
        ê°? ì¸µì˜ ê¸°ìš¸ê¸°ë?? ?‹´??? ?”•?…”?„ˆë¦?(dictionary) ë³??ˆ˜
            grads['W1']??grads['W2']???... ê°? ì¸µì˜ ê°?ì¤‘ì¹˜
            grads['b1']??grads['b2']???... ê°? ì¸µì˜ ?¸?–¥
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """ê¸°ìš¸ê¸°ë?? êµ¬í•œ?‹¤(?˜¤ì°¨ì—­? „?ŒŒë²?).

        Parameters
        ----------
        x : ?…? ¥ ?°?´?„°
        t : ? •?‹µ ? ˆ?´ë¸?
        
        Returns
        -------
        ê°? ì¸µì˜ ê¸°ìš¸ê¸°ë?? ?‹´??? ?”•?…”?„ˆë¦?(dictionary) ë³??ˆ˜
            grads['W1']??grads['W2']???... ê°? ì¸µì˜ ê°?ì¤‘ì¹˜
            grads['b1']??grads['b2']???... ê°? ì¸µì˜ ?¸?–¥
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # ê²°ê³¼ ????¥
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
