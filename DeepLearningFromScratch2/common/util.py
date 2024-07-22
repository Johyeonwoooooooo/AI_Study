# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps=1e-8):
    '''肄붿궗�씤 �쑀�궗�룄 �궛異�

    :param x: 踰≫꽣
    :param y: 踰≫꽣
    :param eps: '0�쑝濡� �굹�늻湲�'瑜� 諛⑹���븯湲� �쐞�븳 �옉��� 媛�
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''�쑀�궗 �떒�뼱 寃��깋

    :param query: 荑쇰━(�뀓�뒪�듃)
    :param word_to_id: �떒�뼱�뿉�꽌 �떒�뼱 ID濡� 蹂��솚�븯�뒗 �뵓�뀛�꼫由�
    :param id_to_word: �떒�뼱 ID�뿉�꽌 �떒�뼱濡� 蹂��솚�븯�뒗 �뵓�뀛�꼫由�
    :param word_matrix: �떒�뼱 踰≫꽣瑜� �젙由ы븳 �뻾�젹. 媛� �뻾�뿉 �빐�떦 �떒�뼱 踰≫꽣媛� ����옣�릺�뼱 �엳�떎怨� 媛��젙�븳�떎.
    :param top: �긽�쐞 紐� 媛쒓퉴吏� 異쒕젰�븷 吏� 吏��젙
    '''
    if query not in word_to_id:
        print('%s(�쓣)瑜� 李얠쓣 �닔 �뾾�뒿�땲�떎.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 肄붿궗�씤 �쑀�궗�룄 怨꾩궛
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 肄붿궗�씤 �쑀�궗�룄瑜� 湲곗���쑝濡� �궡由쇱감�닚�쑝濡� 異쒕젰
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocab_size):
    '''�썝�빂 �몴�쁽�쑝濡� 蹂��솚

    :param corpus: �떒�뼱 ID 紐⑸줉(1李⑥썝 �삉�뒗 2李⑥썝 �꽆�뙆�씠 諛곗뿴)
    :param vocab_size: �뼱�쐶 �닔
    :return: �썝�빂 �몴�쁽(2李⑥썝 �삉�뒗 3李⑥썝 �꽆�뙆�씠 諛곗뿴)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(corpus, vocab_size, window_size=1):
    '''�룞�떆諛쒖깮 �뻾�젹 �깮�꽦

    :param corpus: 留먮춬移�(�떒�뼱 ID 紐⑸줉)
    :param vocab_size: �뼱�쐶 �닔
    :param window_size: �쐢�룄�슦 �겕湲�(�쐢�룄�슦 �겕湲곌�� 1�씠硫� ���源� �떒�뼱 醫뚯슦 �븳 �떒�뼱�뵫�씠 留λ씫�뿉 �룷�븿)
    :return: �룞�떆諛쒖깮 �뻾�젹
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI(�젏蹂� �긽�샇�젙蹂대웾) �깮�꽦

    :param C: �룞�떆諛쒖깮 �뻾�젹
    :param verbose: 吏꾪뻾 �긽�솴�쓣 異쒕젰�븷吏� �뿬遺�
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% �셿猷�' % (100*cnt/total))
    return M


def create_contexts_target(corpus, window_size=1):
    '''留λ씫怨� ���源� �깮�꽦

    :param corpus: 留먮춬移�(�떒�뼱 ID 紐⑸줉)
    :param window_size: �쐢�룄�슦 �겕湲�(�쐢�룄�슦 �겕湲곌�� 1�씠硫� ���源� �떒�뼱 醫뚯슦 �븳 �떒�뼱�뵫�씠 留λ씫�뿉 �룷�븿)
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('�띁�뵆�젆�꽌�떚 �룊媛� 以� ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char,
                 verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 癒몃┸湲��옄
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 臾몄옄�뿴濡� 蹂��솚
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '�삊' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '�삋' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s(�쓣)瑜� 李얠쓣 �닔 �뾾�뒿�땲�떎.' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x
