import random
import numpy as np
from tqdm import tqdm
from utils import *


def train_civilian_model(epochs):
    Q_table = [np.zeros((num_states, num_actions_night)), np.zeros((num_states, num_actions_discuss)), np.zeros((num_states, num_actions_vote))]

    for epoch in tqdm(range(epochs), desc='train civilian model'):
        players = [i for i in range(num_players)]
        agent, mafia_player = random.sample(players, 2)
        phase = 0  # 0 - night, 1 - day discussion, 2 - day vote
        first_round = True

        s_alive = [1 for i in range(num_players)]
        s_accusation = np.zeros((num_players, num_players), dtype=int)
        state = encode_state(s_alive + s_accusation.flatten().tolist(), 2)

        while is_end_state(s_alive, agent, mafia_player) < 0:

            action = None

            # nighttime
            if phase == 0:
                if first_round == True:
                    mafia_action = 2
                    first_round = False
                else:
                    mafia_action = random.randint(0, 2)   # 0 - most accused, most accusing, random
                civilians = [i for i in range(num_players) if s_alive[i] == 1 and i != mafia_player]
                eliminated = random.choice(civilians)
                if mafia_action == 0:
                    temp = np.argmax(np.sum(s_accusation, axis=0))
                    if temp != mafia_player:
                        eliminated = temp
                elif mafia_action == 1:
                    temp = np.argmax(np.sum(s_accusation, axis=1))
                    if temp != mafia_player:
                        eliminated = temp
                s_alive[eliminated] = 0
                if agent == mafia_player:
                    action = mafia_action

            # daytime discuss
            elif phase == 1:
                if np.random.rand() < exploration_prob:
                    not_me = [i for i in range(num_players) if s_alive[i] == 1 and i != agent]
                    to_accuse = random.choice(not_me)
                else:
                    to_accuse = np.argmax(Q_table[phase][state])
                s_accusation = generate_discuss_matrix(num_players, s_alive, agent, to_accuse)
                action = to_accuse

            # daytime vote
            # most accused / most accusing / random
            elif phase == 2:
                if np.random.rand() < exploration_prob:
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(Q_table[phase][state])
                people_alive = [i for i in range(num_players) if s_alive[i] == 1]
                votes = generate_vote_matrix(num_players, s_alive, agent, action)

                elimination = [0 for i in range(num_players)]
                for v in votes:
                    if v == -1:
                        continue
                    if v == 0:
                        temp = np.argmax(np.sum(s_accusation, axis=0))
                        elimination[temp] += 1
                    elif v == 1:
                        temp = np.argmax(np.sum(s_accusation, axis=1))
                        elimination[temp] += 1
                    else:
                        people_alive = [i for i in range(num_players) if s_alive[i] == 1]
                        temp = random.choice(people_alive)
                        elimination[temp] += 1
                    
                eliminated = np.argmax(elimination)
                s_alive[eliminated] = 0

            next_state = encode_state(s_alive + s_accusation.flatten().tolist(), 2)

            if action is None:
                state = next_state
                phase = next_phase(phase)
                continue

            reward = get_reward(s_alive, agent, mafia_player)

            Q_table[phase][state][action] += learning_rate * \
                (reward + discount_factor *
                    np.max(Q_table[phase][next_state]) - Q_table[phase][state][action])

            state = next_state

            phase = next_phase(phase)

    return Q_table


def train_mafia_model(epochs):
    Q_table_mafia = [np.zeros((num_states, num_actions_night)), np.zeros((num_states, num_actions_discuss)), np.zeros((num_states, num_actions_vote))]

    for epoch in tqdm(range(epochs), desc='train mafia model'):
        players = [i for i in range(num_players)]
        agent = random.choice(players)
        mafia_player = agent
        phase = 0  # 0 - night, 1 - day discussion, 2 - day vote
        first_round = True

        s_alive = [1 for i in range(num_players)]
        s_accusation = np.zeros((num_players, num_players), dtype=int)
        state = encode_state(s_alive + s_accusation.flatten().tolist(), 2)

        while is_end_state(s_alive, agent, mafia_player) < 0:

            action = None

            # nighttime
            if phase == 0:
                if first_round == True:
                    mafia_action = 2
                    first_round = False
                elif np.random.rand() < exploration_prob:
                    mafia_action = random.randint(0, 2)   # 0 - most accused, most accusing, random
                else:
                    mafia_action = np.argmax(Q_table_mafia[phase][state])
                civilians = [i for i in range(num_players) if s_alive[i] == 1 and i != mafia_player]
                eliminated = random.choice(civilians)
                if mafia_action == 0:
                    temp = np.argmax(np.sum(s_accusation, axis=0))
                    if temp != mafia_player:
                        eliminated = temp
                elif mafia_action == 1:
                    temp = np.argmax(np.sum(s_accusation, axis=1))
                    if temp != mafia_player:
                        eliminated = temp
                s_alive[eliminated] = 0
                if agent == mafia_player:
                    action = mafia_action

            # daytime discuss
            elif phase == 1:
                if np.random.rand() < exploration_prob:
                    not_me = [i for i in range(num_players) if s_alive[i] == 1 and i != agent]
                    to_accuse = random.choice(not_me)
                else:
                    to_accuse = np.argmax(Q_table_mafia[phase][state])
                s_accusation = generate_discuss_matrix(num_players, s_alive, agent, to_accuse)
                action = to_accuse

            # daytime vote
            # most accused / most accusing / random
            elif phase == 2:
                if np.random.rand() < exploration_prob:
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(Q_table_mafia[phase][state])
                people_alive = [i for i in range(num_players) if s_alive[i] == 1]
                votes = generate_vote_matrix(num_players, s_alive, agent, action)

                elimination = [0 for i in range(num_players)]
                for v in votes:
                    if v == -1:
                        continue
                    if v == 0:
                        temp = np.argmax(np.sum(s_accusation, axis=0))
                        elimination[temp] += 1
                    elif v == 1:
                        temp = np.argmax(np.sum(s_accusation, axis=1))
                        elimination[temp] += 1
                    else:
                        people_alive = [i for i in range(num_players) if s_alive[i] == 1]
                        temp = random.choice(people_alive)
                        elimination[temp] += 1

                eliminated = np.argmax(elimination)
                s_alive[eliminated] = 0

            next_state = encode_state(s_alive + s_accusation.flatten().tolist(), 2)

            if action is None:
                state = next_state
                phase = next_phase(phase)
                continue

            reward = get_reward(s_alive, agent, mafia_player)

            Q_table_mafia[phase][state][action] += learning_rate * \
                (reward + discount_factor *
                    np.max(Q_table_mafia[phase][next_state]) - Q_table_mafia[phase][state][action])

            state = next_state

            phase = next_phase(phase)
    
    return Q_table_mafia


def eval_models(Q_table_civilian, Q_table_mafia, num_games):
    counts = {'mafia':0, 'civilians':0}

    for epoch in tqdm(range(num_games), desc='eval models'):
        players = [i for i in range(num_players)]
        mafia_player = random.choice(players)
        phase = 0  # 0 - night, 1 - day discussion, 2 - day vote
        first_round = True

        s_alive = [1 for i in range(num_players)]
        s_accusation = np.zeros((num_players, num_players), dtype=int)
        state = encode_state(s_alive + s_accusation.flatten().tolist(), 2)

        while is_end_state_test(s_alive, mafia_player) < 0:

            action = None

            # nighttime
            if phase == 0:
                if first_round == True:
                    mafia_action = 2
                    first_round = False
                else:
                    mafia_action = np.argmax(Q_table_mafia[phase][state])   # 0 - most accused, most accusing, random
                civilians = [i for i in range(num_players) if s_alive[i] == 1 and i != mafia_player]
                eliminated = random.choice(civilians)
                if mafia_action == 0:
                    temp = np.argmax(np.sum(s_accusation, axis=0))
                    if temp != mafia_player:
                        eliminated = temp
                elif mafia_action == 1:
                    temp = np.argmax(np.sum(s_accusation, axis=1))
                    if temp != mafia_player:
                        eliminated = temp
                s_alive[eliminated] = 0

            # daytime discuss
            elif phase == 1:
                to_accuse_civilian = np.argmax(Q_table_civilian[phase][state])
                to_accuse_mafia = np.argmax(Q_table_mafia[phase][state])
                s_accusation = generate_discuss_matrix_test(num_players, s_alive, to_accuse_civilian, to_accuse_mafia, mafia_player)

            # daytime vote
            # most accused / most accusing / random
            elif phase == 2:
                action = np.argmax(Q_table_civilian[phase][state])
                action_mafia = np.argmax(Q_table_civilian[phase][state])
                people_alive = [i for i in range(num_players) if s_alive[i] == 1]

                elimination = [0 for i in range(num_players)]
                for i in people_alive:
                    a = action_mafia if i == mafia_player else action
                    if a == 0:
                        temp = np.argmax(np.sum(s_accusation, axis=0))
                        elimination[temp] += 1
                    elif a == 1:
                        temp = np.argmax(np.sum(s_accusation, axis=1))
                        elimination[temp] += 1
                    else:
                        people_alive = [i for i in range(num_players) if s_alive[i] == 1]
                        temp = random.choice(people_alive)
                        elimination[temp] += 1

                eliminated = np.argmax(elimination)
                s_alive[eliminated] = 0

            next_state = encode_state(s_alive + s_accusation.flatten().tolist(), 2)

            state = next_state

            phase = next_phase(phase)

        ending = is_end_state_test(s_alive, mafia_player)

        if ending == 0:
            counts['mafia'] += 1
        elif ending == 1:
            counts['civilians'] += 1

    print("\n")
    print(f"Wins for each team out of {num_games} games: {counts}")



if __name__ == "__main__":

    num_players = 5
    num_phases = 3
    num_states = 2**(num_players + num_players*num_players)
    num_actions_night = num_players
    night_valid = [1, 1, 1, 1, 1]
    night_valid[0] = 0

    bases_state = [2 for i in range(num_players + num_players * num_players)]
    bases_act = [2 for i in range(num_players*num_players)]

    num_actions_night = 3
    num_actions_discuss = num_players
    num_actions_vote = 3

    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 200000
    num_games = 1000

    Q_table_civilian = train_civilian_model(epochs)
    Q_table_mafia = train_mafia_model(epochs)

    eval_models(Q_table_civilian, Q_table_mafia, num_games)
