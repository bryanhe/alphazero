#!/usr/bin/env python3

import os
import tqdm
import math
import torch
import numpy as np
import numba
import time
import threading
import torch.multiprocessing as multiprocessing
multiprocessing = multiprocessing.get_context("spawn")

def init_state(m=6, n=7):
    return np.zeros((3, m, n), np.bool)

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

def random(state):
    return [np.random.choice(legal_moves(s)) for s in state]

# TODO: try to process root in blocks (and allow multidim root)
def run(i, root, heuristic, batch=True, rollouts=1600, alpha=5.0, tau=1.0, verbose=True):
    np.random.seed(i)
    _, m, n = root.shape
    done = set()  # Do not need to have a different copy per game
    moves = {}
    P = {}
    V = {}
    N = {}
    Q_unnormalized = {}
    v = None
    # TODO: tqdm doesn't make sense if not synced
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
            P[h] = p[0, :]
            V[h] = v[0, 0]

        moves[h] = legal_moves(state)
        # get rid of illegal moves from P
        illegal = illegal_moves(state)
        P[h][illegal] = 0
        if moves[h].shape[0] != 0:
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

def mcts(root, heuristic, batch=True, rollouts=1600, alpha=5.0, tau=1.0, verbose=True):

    start = time.time()
    t = time.time()
    root = root.copy()

    # TODO: check that shapes match
    _, m, n = root[0].shape

    # TODO: empty
    # import code; code.interact(local=dict(globals(), **locals()))
    # state_buffer = np.zeros((len(root), *root[0].shape), np.bool)  # TODO: make shared
    # P_buffer = np.zeros((len(root), root[0].shape[2]))
    # V_buffer = np.zeros((len(root), 1))
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

    # TODO: recycle
    state_buffer = multiprocessing.RawArray("b", sum(r.size for r in root))
    P_buffer = multiprocessing.RawArray("d", 7 * len(root))
    V_buffer = multiprocessing.RawArray("d", 1 * len(root))
    barrier = multiprocessing.Barrier(len(root) + 1)

    state = np.frombuffer(state_buffer, dtype=np.bool).reshape(len(root), *root[0].shape)
    P = np.frombuffer(P_buffer, dtype=np.float).reshape(len(root), 7)
    V = np.frombuffer(V_buffer, dtype=np.float).reshape(len(root), 1)

    # for i in range(len(root)):
    #     run(i)

    # TODO: set option to run sequentially (for debugging, and is faster when only a small number of roots)
    print("Setting up memory:", time.time() - t); t = time.time()
    if run.batch is None or run.batch < len(root):
        run.batch = len(root)
        if run.pool is not None:
            pass  # TODO: free old pool
        # TODO: only need to run init if batch
        run.pool = multiprocessing.Pool(processes=len(root), initializer=init, initargs=(barrier, state_buffer, P_buffer, V_buffer, root))
        print("Setting up pool:", time.time() - t); t = time.time()

    gpu = 0.
    ans = run.pool.starmap_async(run, [(i, r, None if batch else heuristic, batch, rollouts, alpha, tau, verbose) for (i, r) in enumerate(root)])
    if batch:
        for i in range(rollouts):
            barrier.wait()
            t = time.time()
            P[:], V[:] = heuristic(state)
            gpu += time.time() - t
            barrier.wait()
        print("gpu:", gpu); t = time.time()
    ans = ans.get()
    print("sync", time.time() - t); t = time.time()
    print("TOTAL", time.time() -  start)
    print("AVE", (time.time() -  start) / len(root))
    return ans
run.batch = None
run.pool = None


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
        move = player[p]([s for (s, d) in zip(state, done) if not d])

        index = 0
        new_move = [None for _ in states]
        for (i, d) in enumerate(done):
            if not d:
                new_move[i] = move[index]
                index += 1
        move = new_move

        for i in range(len(state)):
            moves[i].append(move[i])
        assert(all(d or m in l for (m, l, d) in zip(move, legal, done)))
        state = [next_state(s, m) if not d else s for (s, m, d) in zip(state, move, done)]
        p = 1 - p
    for i in range(len(state)):
        moves[i].append(None)
    if p1 == human or p2 == human:
    # if True:
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

def selfplay():
    EPOCHS = 1000
    BUFFER_SIZE = 1000
    device = "cuda"
    output = "output"
    os.makedirs(output, exist_ok=True)

    model = Dual(5)
    model.to(device)
    model.share_memory()
    model_heuristic = ModelHeuristic(model, device)

    best = Dual(5)
    best.load_state_dict(model.state_dict())

    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0)

    # TODO: reload values

    state_buffer = []
    move_buffer = []
    reward_buffer = []
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        # TODO: select whether to do fast or proper rollouts
        model.eval()  # TODO: also no_grad
        state, move, reward = play(lambda state: mcts(state, basic_heuristic, batch=False), lambda state: mcts(state, basic_heuristic, batch=False), 128)
        print("Epoch #{} playouts took {} seconds.".format(epoch, time.time() - epoch_start))

        t = time.time()
        for (s, m, r) in zip(state, move, reward):
            for i in range(len(s)):
                state_buffer.append(s[i])
                move_buffer.append(m[i] if m[i] is not None else -1)
                reward_buffer.append(r)

        # TODO: cut buffer to desired length

        total_pl = 0.
        total_rl = 0.
        correct_p = 0
        correct_r = 0
        n = 0
        model.train()
        dataset = torch.utils.data.TensorDataset(torch.as_tensor(state_buffer, dtype=torch.float), torch.as_tensor(move_buffer), torch.as_tensor(reward_buffer))
        loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=256, shuffle=True, drop_last=True)
        for (s, m, r) in loader:
            s = s.to(device)
            m = m.to(device)
            r = r.to(device)
            p, v = model(s)
            policy_loss = torch.nn.functional.cross_entropy(p, m, ignore_index=-1)
            reward_loss = torch.nn.functional.mse_loss(v.view(-1), r.float())
            loss = reward_loss + policy_loss
            total_pl += policy_loss.item() * s.shape[0]
            total_rl += reward_loss.item() * s.shape[0]
            correct_r += (v.round().long().view(-1) == r).sum().item()
            correct_p += (torch.argmax(p, dim=1) == m).sum().item()  # TODO: need to fix denominator for this -- not all states have a move
            n += s.shape[0]
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        print("Policy Loss: {}".format(total_pl / n))
        print("Policy Acc:  {}".format(correct_p / n))
        print("Reward Loss: {}".format(total_rl / n))
        print("Reward Acc:  {}".format(correct_r / n))
        print("Epoch #{} training took {} seconds.".format(epoch, time.time() - epoch_start))

        torch.save({"epoch": epoch,
                    "state_buffer": state_buffer,
                    "move_buffer": move_buffer,
                    "reward_buffer": reward_buffer,
                    "model_state_dict": model.state_dict(),
                    "best_state_dict": best.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    }, os.path.join(output, "checkpoint.pt"))

        # TODO: periodically check which model is better

        # Save metadata


def main():

    # play(human, random)
    # play(lambda state: mcts(state, heuristic), human)
    # play(lambda state: mcts(state, heuristic), lambda state: mcts(state, heuristic), 2)

    # TODO: eval and nograd
    selfplay()

    # for i in range(1000):
    #     states, moves, p = play(lambda state: mcts(state, basic_heuristic, batch=False), lambda state: mcts(state, basic_heuristic, batch=False), 128)
    #     # states, moves, p = play(lambda state: mcts(state, model_heuristic), lambda state: mcts(state, basic_heuristic), 128)
    #     # states, moves, p = play(lambda state: mcts(state, basic_heuristic), lambda state: mcts(state, model_heuristic), 128)
    #     # states, moves, p = play(lambda state: mcts(state, model_heuristic), lambda state: mcts(state, model_heuristic), 128)
    #     print("Reward: ", p, flush=True)

if __name__ == "__main__":
    main()

# GPU not blocked
# For looping in python
# Not reusing tree
