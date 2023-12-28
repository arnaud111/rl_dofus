from utils_dofus import ACTIONS, get_player_pos, END_TURN, MAX_PM, MAX_PA, REWARD_END_TURN, REWARD_END_FIGHT, \
    REWARD_MOB, REWARD_MOVE, REWARD_IMPOSSIBLE, WALL, INVO, MONSTER, mob_turn, dict_swap_pos_to_array_index, EMPTY_CELL, \
    UNDEFINED, BOSS, ldv, USABLE_CELL, map_contains_mob, count_pm_to_pos, PLAYER, REWARD_END_TURN_WITHOUT_ACTION


class Environment:

    def do(self, map_state, action):

        clicked = ACTIONS[action]
        player_pos = get_player_pos(map_state.map)
        new_state = map_state.clone()

        if player_pos == -1:
            exit()

        if clicked == END_TURN:
            reward = REWARD_END_TURN
            if new_state.pm == MAX_PM and new_state.pa == MAX_PA:
                reward = REWARD_END_TURN_WITHOUT_ACTION
            new_state.pa = MAX_PA
            new_state.pm = MAX_PM
            new_state.map = mob_turn(new_state.map, invo=False)
            return reward, new_state

        clicked_pos = player_pos[0] + clicked[0], player_pos[1] + clicked[1]

        if clicked_pos not in dict_swap_pos_to_array_index.keys():
            return REWARD_IMPOSSIBLE, new_state

        clicked_index = dict_swap_pos_to_array_index[clicked_pos]

        if new_state.map[clicked_index] == EMPTY_CELL or new_state.map[clicked_index] == WALL or new_state.map[
            clicked_index] == UNDEFINED:
            return REWARD_IMPOSSIBLE, new_state

        elif new_state.map[clicked_index] == MONSTER or new_state.map[clicked_index] == INVO or new_state.map[
            clicked_index] == BOSS:

            if new_state.pa < 3:
                return REWARD_IMPOSSIBLE, new_state

            if not ldv(new_state.map, player_pos, clicked_pos):
                return REWARD_IMPOSSIBLE, new_state

            new_state.map[clicked_index] = USABLE_CELL
            new_state.pa -= 3

            if not map_contains_mob(new_state.map):
                return REWARD_END_FIGHT, new_state
            return REWARD_MOB, new_state

        elif new_state.map[clicked_index] == USABLE_CELL:
            used = count_pm_to_pos(new_state.map, player_pos, clicked_pos, new_state.pm)

            if used == -1:
                return REWARD_IMPOSSIBLE, new_state

            new_state.map[clicked_index] = PLAYER
            new_state.map[dict_swap_pos_to_array_index[player_pos]] = USABLE_CELL
            new_state.pm -= used
            return REWARD_MOVE, new_state
