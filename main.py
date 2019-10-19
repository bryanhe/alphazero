#!/usr/bin/env python3

import tqdm
import math
import torch
import numpy as np
import numba
import time
import threading
import torch.multiprocessing as multiprocessing
multiprocessing = multiprocessing.get_context("spawn")

# TODO: need to terminate game if tie

def print_board(state):
    _, m, n = state.shape
    print(" " + (4 * n - 1) * "-" + " ")
    for i in range(m):
        print("|", end="")
        for j in range(n):
            print(" ", end = '')
            assert(not (state[0, i, j] and state[1, i, j]))
            if state[0, i, j]:
                print("X", end = '')
            elif state[1, i, j]:
                print("O", end = '')
            else:
                print(" ", end = '')
            print(" |", end = '')
        print(" ")
        print(" " + (4 * n - 1) * "-" + " ")
    print(" " + "".join(["{:^4d}".format(i) for i in range(n)]) + " ", flush=True)

def init_state(m=6, n=7):
    return np.zeros((3, m, n), np.bool)

@numba.jit(cache=True, nopython=True)
def score(state):
    # import code; code.interact(local=locals())
    _, m, n = state.shape
    for p in range(2):
        for i in range(m - 3):
            for j in range(n):
                if (state[p, i + 0, j] and
                    state[p, i + 1, j] and
                    state[p, i + 2, j] and
                    state[p, i + 3, j]):
                    return 2 * p - 1

        for i in range(m):
            for j in range(n - 3):
                if (state[p, i, j + 0] and
                    state[p, i, j + 1] and
                    state[p, i, j + 2] and
                    state[p, i, j + 3]):
                    return 2 * p - 1

        for i in range(m - 3):
            for j in range(n - 3):
                if (state[p, i + 0, j + 0] and
                    state[p, i + 1, j + 1] and
                    state[p, i + 2, j + 2] and
                    state[p, i + 3, j + 3]):
                    return 2 * p - 1
                if (state[p, i + 3, j + 0] and
                    state[p, i + 2, j + 1] and
                    state[p, i + 1, j + 2] and
                    state[p, i + 0, j + 3]):
                    return 2 * p - 1

    # state = state[:2, :, :]
    # state = np.concatenate((np.zeros((2, 3, n), np.bool), state, np.zeros((2, 3, n), np.bool)), axis=1)
    # state = np.concatenate((np.zeros((2, m + 6, 3), np.bool), state, np.zeros((2, m + 6, 3), np.bool)), axis=2)
    return 0

# @numba.jit(cache=True, nopython=True)
def legal_moves(state):
    return np.logical_not(state[:2, 0, :].any(0)).nonzero()[0]

def illegal_moves(state):
    return state[:2, 0, :].any(0).nonzero()[0]

@numba.jit(cache=True, nopython=True)
def next_state(state, move):
    state = state.copy()
    p = int(state[2, 0, 0])
    for i in range(state.shape[1] - 1, -1, -1):
        if not (state[0, i, move] or state[1, i, move]):
            state[p, i, move] = True
            break
    state[2, :, :] = np.logical_not(state[2, :, :])
    return state

# @numba.jit(cache=True, nopython=True)
def hash_state(state):
    return state.tobytes()
    # # x = 1
    # h = 0
    # for i in range(state.shape[1]):
    #     for j in range(state.shape[2]):
    #         h += 2 * state[0, i, j] + state[1, i, j]
    #         h *= 3
    #         # x *= 3
    # # print(x)
    # return h

def random(state):
    return [np.random.choice(legal_moves(s)) for s in state]

def run(i, root, heuristic, batch=True, rollouts=1600, alpha=5.0, tau=1.0, verbose=True):
    np.random.seed(i)
    _, m, n = root.shape
    start = time.time()
    done = set()  # Do not need to have a different copy per game
    moves = {}
    P = {}
    V = {}
    N = {}
    Q_unnormalized = {}
    v = None
    for r in tqdm.trange(rollouts, disable=(not verbose or i != 0)):
        ### Selection ###
        state = root.copy()
        visited = []
        depth = 0
        while True:
            h = hash_state(state)
            if h not in N or h in done:  # TODO condition is broken now, since h might get added by another thread
                break
            Q = Q_unnormalized[h] / np.maximum(N[h], 1)
            if not state[2, 0, 0]:
                Q = - Q
            ucb = Q + alpha * math.sqrt(N[h].sum()) * P[h] / (1 + N[h])
            move = max(((N[h][m] == 0) * P[h][m], ucb[m], m) for m in moves[h])[2] # TODO as zip?
            # if depth == 0:
            #     print(ucb)
            #     print(Q)
            #     print(alpha * P[h] * np.sqrt(N[h].sum()) / (1 + N[h]))
            #     print(P[h])
            #     print(N[h])
            #     print()
            visited.append((h, move))
            state = next_state(state, move)
            depth += 1

        # Expand
        if batch:
            state_buffer[i, ...] = state
            barrier.wait()
            barrier.wait()
            P[h] = P_buffer[i, :]
            V[h] = V_buffer[i, 0]
        else:
            p, v = heuristic(np.expand_dims(state, 0))
            P[h] = p[0, :]  # TODO: ???
            V[h] = v[0, 0]

        moves[h] = legal_moves(state)
        # get rid of illegal moves from P
        illegal = illegal_moves(state)
        P[h][illegal] = 0
        P[h] /= P[h].sum()

        s = score(state)
        if s != 0 or moves[h].shape[0] == 0:
            V[h] = s
            done.add(h)

        N[h] = np.zeros(n, np.int)
        Q_unnormalized[h] = np.zeros(n)
        v = V[h]

        # Backup
        for (h, move) in visited:
            Q_unnormalized[h][move] += v
            N[h][move] += 1
    p = N[hash_state(root)] ** (1 / tau)
    p /= p.sum()
    return np.random.choice(7, p=p)

def init(barrier_, state_buffer_, P_buffer_, V_buffer_, root):
    global barrier
    barrier = barrier_
    global state_buffer
    state_buffer = np.frombuffer(state_buffer_, dtype=np.bool).reshape(len(root), *root[0].shape)
    global P_buffer
    P_buffer = np.frombuffer(P_buffer_, dtype=np.float).reshape(len(root), 7)
    global V_buffer
    V_buffer = np.frombuffer(V_buffer_, dtype=np.float).reshape(len(root), 1)

class HeuristicEval(object):
    def __init__(self, heuristic):
        self.heuristic = heuristic
    def __call__(self):
        P_buffer[:], V_buffer[:] = self.heuristic(state_buffer)

def mcts(root, heuristic, batch=True, rollouts=1600, alpha=5.0, tau=1.0, verbose=True):

    start = time.time()
    root = root.copy()

    # TODO: check that shapes match
    _, m, n = root[0].shape

    # TODO: empty
    # import code; code.interact(local=dict(globals(), **locals()))
    # state_buffer = np.zeros((len(root), *root[0].shape), np.bool)  # TODO: make shared
    # P_buffer = np.zeros((len(root), root[0].shape[2]))
    # V_buffer = np.zeros((len(root), 1))
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    state_buffer = multiprocessing.RawArray("b", sum(r.size for r in root))
    P_buffer = multiprocessing.RawArray("d", 7 * len(root))
    V_buffer = multiprocessing.RawArray("d", 1 * len(root))

    state = np.frombuffer(state_buffer, dtype=np.bool).reshape(len(root), *root[0].shape)
    P = np.frombuffer(P_buffer, dtype=np.float).reshape(len(root), 7)
    V = np.frombuffer(V_buffer, dtype=np.float).reshape(len(root), 1)

    heuristic_eval = HeuristicEval(heuristic)
    # barrier = multiprocessing.Barrier(len(root), action=heuristic_eval)
    barrier = multiprocessing.Barrier(len(root) + 1)  # TODO add back action

                
    # print("DEBUG")
    # print_board(root)
    # h = hash_state(root)
    # print(N[hash_state(root)])
    # print(P[hash_state(root)])
    # print(Q_unnormalized[hash_state(root)])
    # Q = Q_unnormalized[h] / N[h]
    # ucb = Q + alpha * P[h] / (1 + N[h])
    # print(ucb)

    # for i in range(len(root)):
    #     run(i)

    # thread = [threading.Thread(target=run, args=(i, )) for i in range(len(root))]
    # for t in thread:
    #     t.start()
    # for t in thread:
    #     t.join()

    print("DEBUG", flush=True)
    t = time.time()
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init, initargs=(barrier, state_buffer, P_buffer, V_buffer, root)) as pool:
    with multiprocessing.Pool(processes=len(root), initializer=init, initargs=(barrier, state_buffer, P_buffer, V_buffer, root)) as pool:
        print("A", time.time() - t); t = time.time()
        # TODO: just return from this
        ans = pool.starmap_async(run, [(i, r, None if batch else heuristic, batch, rollouts, alpha, tau, verbose) for (i, r) in enumerate(root)])
        for i in range(rollouts):
            barrier.wait()
            P[:], V[:] = heuristic(state)
            barrier.wait()
        print("B", time.time() - t); t = time.time()
        ans = ans.get()
        # ans = []
        # for x in [(i, r, heuristic, batch, rollouts, alpha, tau, verbose) for (i, r) in enumerate(root)]:
        #     ans.append(run(*x))
        # return ans
    print("C", time.time() - t); t = time.time()
    print("TOTAL", time.time() -  start)
    print("AVE", (time.time() -  start) / len(root))
    return ans

class _ConvBlock(torch.nn.Module):
    def __init__(self, filters=256):
        super(_ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1)
        self.norm = torch.nn.BatchNorm2d(filters)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class _ResidualBlock(torch.nn.Module):
    def __init__(self, filters=256):
        super(_ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(filters)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(filters)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        x = self.relu2(x)
        return x

class _ValueHead(torch.nn.Module):
    def __init__(self, filters=256, hidden_dim=256):
        super(_ValueHead, self).__init__()
        self.conv = torch.nn.Conv2d(filters, 1, kernel_size=1, stride=1)
        self.norm = torch.nn.BatchNorm2d(1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(7 * 6 * 1, hidden_dim)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

class _PolicyHead(torch.nn.Module):
    def __init__(self, filters=256, channels=2):
        super(_PolicyHead, self).__init__()
        self.conv = torch.nn.Conv2d(filters, channels, kernel_size=1, stride=1)
        self.norm = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(7 * 6 * channels, 7)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Dual(torch.nn.Module):
    def __init__(self, layers, filters=256, policy_channels=2, value_hidden_dim=256):
        super(Dual, self).__init__()
        self.conv = _ConvBlock(filters)
        self.layer = torch.nn.Sequential(*[_ResidualBlock(filters) for _ in range(layers)])
        self.policy = _PolicyHead(filters, policy_channels)
        self.value = _ValueHead(filters, value_hidden_dim)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        p = self.policy(x)
        v = self.value(x)
        return p, v

def human(state):
    print_board(state)

    moves = set(legal_moves(state))
    # print(legal_moves)
    move = None
    while move not in moves:
        try:
            move = int(input("Move: "))
        except ValueError:
            move = None
        except EOFError:
            move = None
            print()
    return move

def play(p1, p2, n=1):
    states = [[] for _ in range(n)]
    moves = [[] for _ in range(n)]
    done = [False for _ in range(n)]
    state = [init_state() for _ in range(n)]
    player = [p1, p2]
    p = 0
    while True:
        for i in range(len(state)):
            if not done[i]:
                states[i].append(state[i])
        sc = [score(i) for i in state]
        legal = [legal_moves(i) for i in state]
        done = [s != 0 or l.shape[0] == 0 for (s, l) in zip(sc, legal)]
        if all(done):
            break
        move = player[p](state)  # TODO: only query moves that arent done
        t = time.time()
        moves.append(move)
        assert(all(m in l for (m, l) in zip(move, legal)))
        state = [next_state(s, m) if not d else s for (s, m, d) in zip(state, move, done)]
        p = 1 - p
        print("ASD", time.time() - t)
    # if p1 == human or p2 == human:
    if True: # p1 == human or p2 == human:
        for ns in state:
            print_board(s)
    return states, moves, sc

def basic_heuristic(state):
    return np.ones((state.shape[0], 7)), np.zeros((state.shape[0], 1))

# https://stackoverflow.com/questions/573569/python-serialize-lexical-closures
class ModelHeuristic(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def __call__(self, state):
        # print(state)
        # P, V = self.model(torch.Tensor(state).to(self.device))
        P, V = self.model(torch.Tensor(state).to(self.device))
        P = torch.softmax(P, 1)
        return P.detach().cpu().numpy(), V.detach().cpu().numpy()

def main():

    # play(human, random)
    # play(lambda state: mcts(state, heuristic), human)
    # play(lambda state: mcts(state, heuristic), lambda state: mcts(state, heuristic), 2)
    # TODO: eval and nograd

    # states, moves, p = play(random, random)
    # asd

    device = "cuda"
    model = Dual(5)
    model.to(device)
    model.eval()
    # # play(lambda state: mcts(state, heuristic), human)
    # import cProfile
    # # cProfile.run("states, moves, p = play(lambda state: mcts(state, heuristic), lambda state:mcts(state, lambda state: (np.ones(7), 0)))")
    # cProfile.run("states, moves, p = play(lambda state: mcts(state, lambda state: (np.ones(7), 0)), lambda state:mcts(state, lambda state: (np.ones(7), 0)))")
    # asd
    model.share_memory()
    model_heuristic = ModelHeuristic(model, device)

    for i in range(1000):
        # states, moves, p = play(lambda state: mcts(state, heuristic), random)
        # states, moves, p = play(lambda state: mcts(state, heuristic), lambda state:mcts(state, lambda state: (np.ones(7), 0)))
        # states, moves, p = play(lambda state: mcts(state, basic_heuristic), lambda state: mcts(state, basic_heuristic), 100)
        states, moves, p = play(lambda state: mcts(state, model_heuristic), lambda state: mcts(state, model_heuristic), 128)
        # states, moves, p = play(lambda state: mcts(state, basic_heuristic), lambda state: mcts(state, basic_heuristic), 1)
        # states, moves, p = play(lambda state: mcts(state, heuristic), lambda state: mcts(state, heuristic), 1000)
        print("Reward: ", p, flush=True)

if __name__ == "__main__":
    main()

# GPU not blocked
# For looping in python
# Not reusing tree
