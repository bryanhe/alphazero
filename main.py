#!/usr/bin/env python3

import torch
import numpy as np
import numba

def print_board(state):
    _, m, n = state.shape
    print(" " + (4 * n - 1) * "-" + " ")
    for i in range(m):
        print("|", end="")
        for j in range(n):
            print(" ", end = '')
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
    # x = 1
    h = 0
    for i in range(state.shape[1]):
        for j in range(state.shape[2]):
            h += 2 * state[0, i, j] + state[1, i, j]
            h *= 3
            # x *= 3
    # print(x)
    return h

def random(state):
    return np.random.choice(legal_moves(state))

import tqdm
def mcts(state, heuristic, rollouts=1000, alpha=1.0, tau=1.0):
    root = state.copy()

    _, m, n = state.shape
    P = {}
    V = {}
    N = {}
    done = set()
    moves = {}
    Q_unnormalized = {}
    for r in tqdm.trange(rollouts):
        # print("rollout {}".format(r))
        ### Selection ###
        visited = []
        state = root.copy()
        depth = 0
        while True:
            # print_board(state)
            h = hash_state(state)
            # print("hash", h)
            if h not in P or h in done:
                break
            Q = Q_unnormalized[h] / N[h]
            if not state[2, 0, 0]:
                Q = - Q
            ucb = Q + alpha * P[h] * np.sqrt(N[h].sum()) / (1 + N[h])
            move = sorted((N[h][m] == 0, P[h][m], ucb[m], m) for m in moves[h])[-1][3]
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
        P[h], V[h] = heuristic(state)
        # get rid of illegal moves from P
        moves[h] = legal_moves(state)
        illegal = illegal_moves(state)
        P[h][illegal] = 0
        P[h] /= P[h].sum()
        s = score(state)
        if s != 0:
            V[h] = s
            done.add(h)
        N[h] = np.zeros(n, np.int)
        Q_unnormalized[h] = np.zeros(n)
        v = V[h]

        # Backup
        for (h, move) in visited:
            Q_unnormalized[h][move] += v
            N[h][move] += 1
            
    # print("DEBUG")
    # print_board(root)
    # h = hash_state(root)
    # print(N[hash_state(root)])
    # print(P[hash_state(root)])
    # print(Q_unnormalized[hash_state(root)])
    # Q = Q_unnormalized[h] / N[h]
    # ucb = Q + alpha * P[h] / (1 + N[h])
    # print(ucb)
    # asd
    # import code; code.interact(local=locals())
    return np.argmax(N[hash_state(root)]) # TODO: temperature

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

def play(p1, p2):
    states = []
    moves = []
    state = init_state()
    player = [p1, p2]
    p = 0
    while True:
        states.append(state)
        s = score(state)
        if s != 0:
            break
        move = player[p](state)
        moves.append(move)
        assert(move in legal_moves(state))
        state = next_state(state, move)
        p = 1 - p
    if p1 == human or p2 == human:
        print_board(state)
    return states, moves, p

def main():

    # play(human, random)
    # play(human, lambda state: mcts(state, None))
    # heuristic = lambda state: (np.ones(7), 0)
    # play(lambda state: mcts(state, heuristic), human)
    # TODO: eval and nograd
    model = Dual(1)
    def heuristic(state):
        P, V = model(torch.Tensor(state).unsqueeze(0))
        return P.detach().numpy()[0, :], V.item()
    play(lambda state: mcts(state, heuristic), human)

    for i in range(10):
        states, moves, p = play(lambda state: mcts(state, heuristic), random)
        print(p)

if __name__ == "__main__":
    main()
