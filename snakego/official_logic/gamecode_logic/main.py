import random
import traceback
from typing import Optional, List

from logic.communication import *
from logic.infrastructure import Context, History, GameConfig
from logic.operate import Controller
from logic.result import Result, ResultType, settle_round
from logic.constants import ITEM_EXPIRE_TIME

def try_parse_int(obj, key, cfg):
    try:
        cfg.__dict__[key] = int(obj["config"][key])
    except:
        pass


def to_B(x):
    return x.to_bytes(1, byteorder='big', signed=False)


def to_I(x):
    return x.to_bytes(2, byteorder='big', signed=True)


class Logic:
    replayPath: Optional[str] = None
    ctx: Optional[Context] = None
    controller: Optional[Controller] = None
    history: History = History()
    player_list: [int, int] = [0, 0]

    def try_resolve_init_info(self, msg):
        logging.info("Try to init game")
        try:
            player_list = msg["player_list"]
            player_num = msg["player_num"]
            replay = msg["replay"]

            config = GameConfig()
            for key in ['length', 'width', 'max_round', 'random_seed']:
                try_parse_int(msg, key, config)
            self.history.game_config = config

            assert type(player_list) == list, "'player_list' is not a list"
            assert len(player_list) == 2, "Player number should be 2"
            assert player_num == 2, "Player number should be 2"

            if player_list[0] == 0 and player_list[1] == 0:
                raise RuntimeError("No ready player")
            elif player_list[0] == 0:
                self.write_gameover_message(1, [0, 0], type=ErrorType.RE, player_error="Player Not Ready")
            elif player_list[1] == 0:
                self.write_gameover_message(0, [0, 0], type=ErrorType.RE, player_error="Player Not Ready")
            else:
                def player_desc(idx: int):
                    return "Local AI" if player_list[idx] == 1 else "Web Player"

                logging.info("Players: [%s, %s]", player_desc(0), player_desc(1))
                self.player_list[0] = player_list[0]
                self.player_list[1] = player_list[1]

            # assert os.access(os.path.dirname(replay), os.W_OK), f"Replay file path [{replay}] is invalid"
            self.replayPath = replay
            logging.info("Replay file will be saved to %s", replay)
            return True
        except KeyError as err:
            logging.error("Missing required field '%s'", str(err))
        except AssertionError as err:
            logging.error(str(err))
        return False

    def write_gameover_message(self, win_side: int, scores: List[int], type: Optional[ErrorType] = None, player_error: Optional[str] = None,
                               internal_error: Optional[str] = None):
        state = [0, 0]
        if internal_error:
            res = Result(ResultType.INTERNAL_ERROR, -1, scores, err=internal_error)
        elif player_error:
            if type == ErrorType.RE or type == ErrorType.TLE:
                res = Result(ResultType.PLAYER_ERROR, win_side, scores, err=player_error)
            elif type == ErrorType.IA:
                res = Result(ResultType.ILLEGAL_ACTION, win_side, scores, err=player_error)
                state[1 - win_side] = 1
            elif type == ErrorType.PE or type == ErrorType.OLE:
                res = Result(ResultType.INVALID_FORMAT, win_side, scores, err=player_error)
                state[1 - win_side] = 1
        else:
            res = Result(ResultType.NORMAL, win_side, scores, err=None)
            send_normal_round_msg(self.history.game_config.max_round * 2 + 1, [], [], [])
        self.history.end_info = res
        self.history.dump(self.replayPath)
        for i in range(2):
            if self.player_list[i] == 1:
                # send_direct_msg_to_player(i, to_B(17) + to_B(res.type.value) + to_B(res.winner) + to_I(scores[0]) + to_I(scores[1]))
                pass
            elif self.player_list[i] == 2:
                send_direct_msg_to_human(i, res)
        send_game_over(scores, state)

    def run(self):
        logging.info("THUAC2022 Logic")

        while not self.try_resolve_init_info(listen()):
            pass

        random.seed(self.history.game_config.random_seed)
        self.ctx = Context(self.history.game_config)
        self.controller = Controller(self.ctx)
        logging.info("Game inited")

        running = True
        score = [0, 0]
        self.history.item_list = self.ctx.game_map.item_list.copy()
        for i in range(2):
            # Note the naming of width/height/length
            if self.player_list[i] == 1:
                msg = to_B(self.history.game_config.length) + to_B(self.history.game_config.width)
                msg += to_I(self.history.game_config.max_round) + to_B(i)
                send_direct_msg_to_player(i, msg)
            elif self.player_list[i] == 2:
                send_direct_msg_to_human(i, {
                    'width': self.history.game_config.length,
                    'height': self.history.game_config.width,
                    'round_count': self.history.game_config.max_round,
                    'player': i
                })

        msg = to_B(16) + to_I(len(self.ctx.game_map.item_list))
        for item in self.ctx.game_map.item_list:
            msg += to_B(item.x) + to_B(item.y) + to_B(item.type) + to_I(item.time) + to_I(item.param)
        for i in range(2):
            if self.player_list[i] == 1:
                send_direct_msg_to_player(i, msg)
            elif self.player_list[i] == 2:
                send_direct_msg_to_human(i, self.ctx.game_map.item_list)
        self.ctx.turn = 1
        game_round = 0
        player_operations = []
        try:
            while running:
                round_info = self.controller.round_preprocess(ITEM_EXPIRE_TIME)
                for i in range(2):
                    if self.player_list[i] == 2:
                        send_direct_msg_to_human(i, round_info)
                self.history.round_info.append(round_info)
                for current in range(2):
                    game_round = game_round + 1
                    logging.info("Round %d started", game_round)
                    logging.info("Reading player %d operation", current)
                    self.controller.round_init()
                    if self.controller.next_snake == -1:
                        self.history.operations.append([])
                        player_operations = []
                        self.controller.next_player()
                        continue

                    send_normal_round_msg(game_round, [current], [], [],
                                          3 if self.player_list[current] == 1 else 600)
                    if game_round > 1:
                        for i in range(2):
                            if self.player_list[i] == 2:
                                send_direct_msg_to_human(i, {'next_player': current})
                    while True:
                        op = read_operation(current, self.player_list[current], game_round)
                        logging.info(str(op))
                        logging.info("Current Player: %d %d %d", current, self.ctx.current_player,
                                     self.controller.player)
                        processed = self.controller.apply(op)
                        player_operations.append(processed)
                        if op.type == 1:
                            msg = to_B(op.direction + 1)
                        else:
                            msg = to_B(op.type + 3)
                        for i in range(2):
                            if self.player_list[i] == 2:
                                send_direct_msg_to_human(i, processed)
                            elif self.player_list[i] == 1:
                                send_direct_msg_to_player(i, msg)
                        if self.controller.next_snake == -1:
                            self.history.operations.append(player_operations.copy())
                            player_operations = []
                            self.controller.next_player()
                            break
                running, score = settle_round(self.ctx)
                # TODO: Deal with a tie
            self.write_gameover_message(0 if score[0] >= score[1] else 1, score)
        except PlayerError as err:
            logging.error(err)
            score[err.player] = -100
            self.write_gameover_message(1 - err.player, score, type=err.type, player_error=err.err_msg)
        except Exception as err:
            logging.exception(err)
            self.write_gameover_message(-1, score, internal_error=traceback.format_exc().replace('"', '\\"'))


if __name__ == '__main__':
    sys.setrecursionlimit(2000)
    # Judger use stdin and stdout. So we use stderr to display log.
    logging.basicConfig(format='%(levelname)s:[Logic.%(module)s:%(lineno)d]: %(message)s', stream=sys.stderr,
                        level=logging.ERROR)
    Logic().run()
