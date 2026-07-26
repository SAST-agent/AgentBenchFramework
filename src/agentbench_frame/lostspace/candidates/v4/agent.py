import enum
import sys
import json
import copy
from enum import Enum

class STATUS(Enum):
    ALIVE = 0
    DEAD = 1
    ESCAPED = 2
    SKIPPED = 3

import random
class Node:
    def __init__(self):
        self.trap = None
        self.drop = [] #掉落物
        self.edges = [] #其中元素为长度为2的数组，第一位为1-8，分别代表从上开始顺时针的八个方向，第二位无意义（本来是门）
        self.able = 1 # 缩圈，1可用，0不可用
        self.type = 0 # 区域类型，0为普通区域，1为物资点, 2为密钥机，3为电梯，4为逃生舱
        pass


class Map:
    #!!!注意Map中的坐标是zyx的顺序
    node = [] # 用三维数组储存节点，从左到右依次为层数（0，1，2），横坐标（0-14），纵坐标（0-14）
    count = [] #三维数组储存高级物资点的刷新倒计时数，同上表示高级物资点的坐标
    circle = -1 #记录缩圈次数
    #初始化各节点的边及区域类型（边中不包含电梯代表的边，可通过区域类型实现）
    def __init__(self):
        #初始化各点的边
        for z in range(3):
            self.node.append([])
            self.count.append([])
            for x in range(7):
                self.node[z].append([])
                self.count[z].append([])
                for y in range(7):
                    self.node[z][x].append(Node())
                    self.count[z][x].append(None)
        self.node[0][0][3].edges = [[3, 1]]
        self.node[0][1][3].edges = [[1, 0], [3, 0], [5, 0], [7, 1]]
        self.node[0][1][2].edges = [[1, 0], [4, 0]]
        self.node[0][1][4].edges = [[2, 0], [5, 0]]
        self.node[0][2][5].edges = [[3, 0], [6, 0]]
        self.node[0][2][4].edges = [[2, 0], [4, 0]]
        self.node[0][2][3].edges = [[3, 0], [7, 0]]
        self.node[0][2][2].edges = [[2, 0], [4, 0]]
        self.node[0][2][1].edges = [[3, 0], [8, 0]]
        self.node[0][3][5].edges = [[3, 0], [4, 0], [6, 0], [7, 0]]
        self.node[0][3][3].edges = [[2, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]
        self.node[0][3][1].edges = [[2, 0], [3, 0], [7, 0], [8, 0]]
        self.node[0][4][5].edges = [[4, 0], [7, 0]]
        self.node[0][4][4].edges = [[6, 0], [8, 0]]
        self.node[0][4][3].edges = [[3, 0], [7, 0]]
        self.node[0][4][2].edges = [[6, 0], [8, 0]]
        self.node[0][4][1].edges = [[2, 0], [7, 0]]
        self.node[0][5][4].edges = [[5, 0], [8, 0]]
        self.node[0][5][3].edges = [[1, 0], [3, 1], [5, 0], [7, 0]]
        self.node[0][5][2].edges = [[1, 0], [6, 0]]
        self.node[0][6][3].edges = [[7, 1]]
        self.node[1][0][6].edges = [[3, 0], [5, 0]]
        self.node[1][0][5].edges = [[1, 0], [3, 0]]
        self.node[1][0][1].edges = [[3, 0], [5, 0]]
        self.node[1][0][0].edges = [[1, 0], [3, 0]]
        self.node[1][1][6].edges = [[3, 1], [5, 0], [7, 0]]
        self.node[1][1][5].edges = [[1, 0], [3, 0], [4, 0], [5, 0], [7, 0]]
        self.node[1][1][4].edges = [[1, 0], [5, 0], [7, 0]]
        self.node[1][1][3].edges = [[1, 0], [5, 0]]
        self.node[1][1][2].edges = [[1, 0], [5, 0], [7, 0]]
        self.node[1][1][1].edges = [[1, 0], [2, 0], [3, 0], [5, 0], [7, 0]]
        self.node[1][1][0].edges = [[1, 0], [3, 1], [7, 0]]
        self.node[1][2][6].edges = [[3, 0], [7, 1]]
        self.node[1][2][5].edges = [[3, 0], [7, 0]]
        self.node[1][2][4].edges = [[2, 0], [3, 0], [5, 0], [8, 0]]
        self.node[1][2][3].edges = [[1, 0], [5, 0]]
        self.node[1][2][2].edges = [[1, 0], [3, 0], [4, 0], [6, 0]]
        self.node[1][2][1].edges = [[3, 0], [7, 0]]
        self.node[1][2][0].edges = [[3, 0], [7, 1]]
        self.node[1][3][6].edges = [[3, 0], [5, 0], [7, 0]]
        self.node[1][3][5].edges = [[1, 0], [3, 0], [4, 0], [6, 0], [7, 0]]
        self.node[1][3][4].edges = [[3, 0], [7, 0]]
        self.node[1][3][2].edges = [[3, 0], [7, 0]]
        self.node[1][3][1].edges = [[2, 0], [3, 0], [5, 0], [7, 0], [8, 0]]
        self.node[1][3][0].edges = [[1, 0], [3, 0], [7, 0]]
        self.node[1][4][6].edges = [[3, 1], [7, 0]]
        self.node[1][4][5].edges = [[3, 0], [7, 0]]
        self.node[1][4][4].edges = [[2, 0], [5, 0], [7, 0], [8, 0]]
        self.node[1][4][3].edges = [[1, 0], [5, 0]]
        self.node[1][4][2].edges = [[1, 0], [4, 0], [6, 0], [7, 0]]
        self.node[1][4][1].edges = [[3, 0], [7, 0]]
        self.node[1][4][0].edges = [[3, 1], [7, 0]]
        self.node[1][5][6].edges = [[3, 0], [5, 0], [7, 1]]
        self.node[1][5][5].edges = [[1, 0], [3, 0], [5, 0], [6, 0], [7, 0]]
        self.node[1][5][4].edges = [[1, 0], [3, 0], [5, 0]]
        self.node[1][5][3].edges = [[1, 0], [5, 0]]
        self.node[1][5][2].edges = [[1, 0], [3, 0], [5, 0]]
        self.node[1][5][1].edges = [[1, 0], [3, 0], [5, 0], [7, 0], [8, 0]]
        self.node[1][5][0].edges = [[1, 0], [3, 0], [7, 1]]
        self.node[1][6][6].edges = [[5, 0], [7, 0]]
        self.node[1][6][5].edges = [[1, 0], [7, 0]]
        self.node[1][6][1].edges = [[5, 0], [7, 0]]
        self.node[1][6][0].edges = [[1, 0], [7, 0]]
        self.node[2][0][6].edges = [[3, 0], [4, 0], [5, 0]]
        self.node[2][0][5].edges = [[1, 0], [4, 0]]
        self.node[2][0][4].edges = [[2, 0], [5, 0]]
        self.node[2][0][3].edges = [[1, 0], [2, 0], [4, 0], [5, 0]]
        self.node[2][0][2].edges = [[1, 0], [4, 0]]
        self.node[2][0][1].edges = [[2, 0], [5, 0]]
        self.node[2][0][0].edges = [[1, 0], [2, 0], [3, 0]]
        self.node[2][1][6].edges = [[4, 0], [7, 0]]
        self.node[2][1][5].edges = [[2, 0], [4, 0], [6, 0], [8, 0]]
        self.node[2][1][4].edges = [[2, 0], [6, 0], [8, 0]]
        self.node[2][1][2].edges = [[4, 0], [6, 0], [8, 0]]
        self.node[2][1][1].edges = [[2, 0], [4, 0], [6, 0], [8, 0]]
        self.node[2][1][0].edges = [[2, 0], [7, 0]]
        self.node[2][2][6].edges = [[3, 0], [6, 0]]
        self.node[2][2][5].edges = [[2, 0], [6, 0], [8, 0]]
        self.node[2][2][4].edges = [[4, 0], [8, 0]]
        self.node[2][2][3].edges = [[2, 0], [4, 0]]
        self.node[2][2][2].edges = [[2, 0], [6, 0]]
        self.node[2][2][1].edges = [[4, 0], [6, 0], [8, 0]]
        self.node[2][2][0].edges = [[3, 0], [8, 0]]
        self.node[2][3][6].edges = [[3, 0], [4, 0], [5, 1], [6, 0], [7, 0]]
        self.node[2][3][5].edges = [[1, 1], [5, 0]]
        self.node[2][3][4].edges = [[1, 0], [4, 0], [6, 0]]
        self.node[2][3][3].edges = [[2, 0], [4, 0], [6, 0], [8, 0]]
        self.node[2][3][2].edges = [[2, 0], [5, 0], [8, 0]]
        self.node[2][3][1].edges = [[1, 0], [5, 1]]
        self.node[2][3][0].edges = [[1, 1], [2, 0], [3, 0], [7, 0], [8, 0]]
        self.node[2][4][6].edges = [[4, 0], [7, 0]]
        self.node[2][4][5].edges = [[2, 0], [4, 0], [8, 0]]
        self.node[2][4][4].edges = [[2, 0], [6, 0]]
        self.node[2][4][3].edges = [[6, 0], [8, 0]]
        self.node[2][4][2].edges = [[4, 0], [8, 0]]
        self.node[2][4][1].edges = [[2, 0], [4, 0], [6, 0]]
        self.node[2][4][0].edges = [[2, 0], [7, 0]]
        self.node[2][5][6].edges = [[3, 0], [6, 0]]
        self.node[2][5][5].edges = [[2, 0], [4, 0], [6, 0], [8, 0]]
        self.node[2][5][4].edges = [[2, 0], [4, 0], [8, 0]]
        self.node[2][5][2].edges = [[2, 0], [4, 0], [6, 0]]
        self.node[2][5][1].edges = [[2, 0], [4, 0], [6, 0], [8, 0]]
        self.node[2][5][0].edges = [[3, 0], [8, 0]]
        self.node[2][6][6].edges = [[5, 0], [6, 0], [7, 0]]
        self.node[2][6][5].edges = [[1, 0], [6, 0]]
        self.node[2][6][4].edges = [[5, 0], [8, 0]]
        self.node[2][6][3].edges = [[1, 0], [5, 0], [6, 0], [8, 0]]
        self.node[2][6][2].edges = [[1, 0], [6, 0]]
        self.node[2][6][1].edges = [[5, 0], [8, 0]]
        self.node[2][6][0].edges = [[1, 0], [7, 0], [8, 0]]
        self.node[1][3][6].type = 1
        self.node[1][3][0].type = 1
        self.node[1][2][3].type = 1
        self.node[1][4][3].type = 1
        self.node[2][3][6].type = 1
        self.node[2][3][0].type = 1
        self.node[2][2][4].type = 1
        self.node[2][2][3].type = 1
        self.node[2][2][2].type = 1
        self.node[2][4][4].type = 1
        self.node[2][4][3].type = 1
        self.node[2][4][2].type = 1
        self.node[0][0][3].type = 3
        self.node[0][3][1].type = 3
        self.node[0][3][5].type = 3
        self.node[0][6][3].type = 3
        self.node[1][3][1].type = 3
        self.node[1][3][5].type = 3
        self.node[2][0][3].type = 3
        self.node[2][3][1].type = 3
        self.node[2][3][5].type = 3
        self.node[2][6][3].type = 3
        self.node[0][3][3].type = 4
        self.node[1][0][0].type = 2
        self.node[1][0][6].type = 2
        self.node[1][6][0].type = 2
        self.node[1][6][6].type = 2
        self.node[2][0][0].type = 2
        self.node[2][0][6].type = 2
        self.node[2][6][0].type = 2
        self.node[2][6][6].type = 2
        #地图初始化

    #区域缩圈（circle自动控制缩圈区域）
    def area_circle(self):
        if self.circle == 0 :#缩圈仓库
            self.node[1][2][4].able = 0
            self.node[1][2][3].able = 0
            self.node[1][2][2].able = 0
            self.node[1][3][4].able = 0
            self.node[1][3][3].able = 0
            self.node[1][3][2].able = 0
            self.node[1][4][4].able = 0
            self.node[1][4][3].able = 0
            self.node[1][4][2].able = 0
        if self.circle == 1 :#通风管
            self.node[2][1][4].able = 0
            self.node[2][2][5].able = 0
            self.node[2][1][2].able = 0
            self.node[2][2][1].able = 0
            self.node[2][4][5].able = 0
            self.node[2][5][4].able = 0
            self.node[2][4][1].able = 0
            self.node[2][5][2].able = 0
        if self.circle == 2 :#中央主机
            self.node[0][2][4].able = 0
            self.node[0][3][4].able = 0
            self.node[0][4][4].able = 0
            self.node[0][2][2].able = 0
            self.node[0][3][2].able = 0
            self.node[0][4][2].able = 0
        if self.circle == 3 :#医疗区
            self.node[1][2][6].able = 0
            self.node[1][3][6].able = 0
            self.node[1][4][6].able = 0
            self.node[1][2][0].able = 0
            self.node[1][3][0].able = 0
            self.node[1][4][0].able = 0
        if self.circle == 4 :#机库
            self.node[2][2][3].able = 0
            self.node[2][3][4].able = 0
            self.node[2][3][3].able = 0
            self.node[2][3][2].able = 0
            self.node[2][4][3].able = 0

    def elevator_beside(self, z, x, y, z1, x1, y1):
        x,y=y,x
        x1,y1=y1,x1
        dir = -1
        if z!=z1:
            if x != x1:
                return False
            if y != y1:
                return False
            if self.node[z][x][y].type == 3:
                return True
            else:
                return False
        if x == x1 and y == y1:
            return True
        if x-x1 == 0 and y-y1 == 1:
            dir = 1
        elif x-x1 == 1 and y-y1 == 1:
            dir = 2
        elif x-x1 == 1 and y-y1 == 0:
            dir = 3
        elif x-x1 == 1 and y-y1 == -1:
            dir = 4
        elif x-x1 == 0 and y-y1 == -1:
            dir = 5
        elif x-x1 == -1 and y-y1 == -1:
            dir = 6
        elif x-x1 == -1 and y-y1 == 0:
            dir = 7
        elif x-x1 == -1 and y-y1 == 1:
            dir = 8
        else:
            return False
        for i in self.node[z1][x1][y1].edges:
            if i[0] == dir:
                return True
        return False

    def beside(self, z, x, y, z1, x1, y1):
        x,y=y,x
        x1,y1=y1,x1
        if z!=z1:
            return False
        if x == x1 and y == y1:
            return True
        if x-x1 == 0 and y-y1 == 1:
            dir = 1
        elif x-x1 == 1 and y-y1 == 1:
            dir = 2
        elif x-x1 == 1 and y-y1 == 0:
            dir = 3
        elif x-x1 == 1 and y-y1 == -1:
            dir = 4
        elif x-x1 == 0 and y-y1 == -1:
            dir = 5
        elif x-x1 == -1 and y-y1 == -1:
            dir = 6
        elif x-x1 == -1 and y-y1 == 0:
            dir = 7
        elif x-x1 == -1 and y-y1 == 1:
            dir = 8
        else:
            return False
        for i in self.node[z1][x1][y1].edges:
            if i[0] == dir:
                return i[1] == 0 or i[2] == 0
        return False


class Tool:  # 工具类,存储工具信息,包括陷阱和医疗包
    def __init__(self) -> None:
        self.landmine_number = []  # 二元组,分别为未放置地雷个数、放置地雷个数
        self.landmine_pos = []  # 地雷的放置位置
        self.sticky_number = []
        self.sticky_pos = []
        self.kit = 0  # 医疗包个数
        self.transport = 0  #瞬移个数


class Player:  # 玩家类,存储该玩家的相关信息
    def __init__(self) -> None:
        self.id = -1  # 玩家id
        self.status = STATUS.ALIVE  # 玩家状态
        self.hp = 200  # 玩家血量
        self.keys = []  # 拿到的钥匙
        self.tools = Tool()  # 工具栏
        self.pos = [-1,-1,-1]   #玩家位置
        self.spawn_pos = []     #玩家出生点位置
        self.last_use_check = -5


class View:
    def __init__(self) -> None:
        self.nodes = []     #视野中的点集合


class AIClient:
    def __init__(self) -> None:
        self.state = -1  # 大回合数
        self.player = Player()  # 玩家信息
        self.others = [Player() for _ in range(3)]  # 其他玩家信息
        self.view = View()  # 玩家视野
        self.root = {}  #接收通信消息
        self.map = Map()

    def receive_data(self):
        """
        接收信息，返回值为字典
        """
        read_buffer = sys.stdin.buffer
        data_len = int(str(read_buffer.read(4), encoding = "utf-8"))
        data = read_buffer.read(data_len)
        self.root = json.loads(str(data, "utf-8"))

    def convert_to_bytes(self, data_str):
        """
        传输数据的时候加数据长度作为数据头
        """
        message_len = len(data_str)
        message = message_len.to_bytes(4, byteorder="big", signed=True)
        message += bytes(data_str, encoding="utf-8")
        return message

    def send_opt(self, data):
        """
        发送自己的操作，data为字典
        """
        sys.stdout.buffer.write(self.convert_to_bytes(json.dumps(data)))
        sys.stdout.flush()

    def init_game(self):
        """
        整局游戏开始时接收消息的处理
        """
        self.player.id = self.root["id"]
        self.player.spawn_pos = self.root["birth_pos"]
        self.player.spawn_pos.append(1)
        self.player.pos = self.player.spawn_pos

    def start_turn(self):
        """
        本玩家回合开始时接收消息的处理
        """
        self.state = self.root["state"]
        if self.root["inturn"] != self.player.id:
            return
        self.player.status = self.root["status"]
        self.player.hp = self.root["hp"]
        self.player.keys = self.root["keys"]
        self.player.tools.landmine_number = [
            self.root["tools"]["LandMine"][0],
            self.root["tools"]["LandMine"][1],
        ]
        self.player.tools.landmine_pos = self.root["tools"]["LandMine"][2:]
        self.player.tools.sticky_number = [
            self.root["tools"]["Sticky"][0],
            self.root["tools"]["Sticky"][1],
        ]
        self.player.tools.sticky_pos = self.root["tools"]["Sticky"][2:]
        self.player.tools.kit = self.root["tools"]["Kit"]
        self.player.tools.transport = self.root["tools"]["Transport"]
        self.root["others"].sort(key=lambda player:player["player_id"])
        for index in range(3):
            self.others[index].id = self.root["others"][index]["player_id"]
            self.others[index].status = self.root["others"][index]["status"]
            self.others[index].keys = self.root["others"][index]["keys"]
            self.others[index].hp = self.root["others"][index]["hp"]
        if self.player.status == STATUS.ALIVE.value:
            self.play()         #选手进行操作
            self.end_turn()
        else:
            # ESCAPED (done), IN_CAPSULE (escape countdown, status 4), or
            # SKIPPED/DEAD: still must ack the turn so the logic advances.
            self.end_turn()
    
    def in_turn(self):
        """
        本玩家回合内接收消息的处理
        """
        if self.root["type"] == "other_death":
            index = self.player_id_to_player_index(self.root["playerid"])
            self.others[index].hp = 0
            self.others[index].status = STATUS.DEAD
        elif self.root["type"] == "getkey":
            key = self.root["key"]
            if key not in self.player.keys:
                self.player.keys.append(key)
        elif self.root["type"] == "escaped":
            self.player.status = STATUS.ESCAPED
        elif self.root["type"] == "death":
            self.player.status = STATUS.DEAD
            self.player.hp = 0
            if self.root["box"]:
                node_index = self.view.nodes[self.pos_to_node_index(self.player.pos)]
                self.view.nodes[node_index].interprops.append("Box")
            self.player.pos = self.player.spawn_pos     #玩家死亡，回到出生点待命

    def off_turn(self):
        """
        本玩家回合外接收消息的处理
        """
        if self.root["content"][0] == "died":
            player_index = self.player_id_to_player_index(self.root["content"][1])
            self.others[player_index].hp = 0
            self.others[player_index].status = STATUS.DEAD
        elif self.root["content"][0] == "escaped":
            player_index = self.player_id_to_player_index(self.root["playerid"])
            self.others[player_index].status = STATUS.ESCAPED
        elif self.root["content"][0] == "see":
            if self.root["content"][1] == "pos_update":
                new_pos = self.root["content"][2]
                player_index = self.player_id_to_player_index(self.root["content"][3])
                self.others[player_index].pos = new_pos
            elif self.root["content"][1] == "attack":
                return  #不需要更新信息
            elif self.root["content"][1] == "regenerate":
                player_index = self.player_id_to_player_index(self.root["content"][3])
                self.others[player_index].status == STATUS.ALIVE
            elif self.root["content"][1] == "interprops_status_update":
                if self.root["content"][3] == "Door":
                    edge_pos = self.root["content"][2]
                    for index,edge in enumerate(self.view.edges):
                        if (edge.stpos == edge_pos[0] and edge.edpos == edge_pos[1]) or (edge.stpos == edge_pos[1] and edge.edpos == edge_pos[0]):
                            self.view.edges[index].is_open = True if self.root["content"][4] == "open" else False
        elif self.root["content"][0] == "getkey":
            player_index = self.player_id_to_player_index(self.root["playerid"])
            for key in self.root["content"][1]:
                if key not in self.others[player_index].keys:
                    self.others[player_index].keys.append(key)
        elif self.root["content"][0] == "hp_update":
            self.player.hp = self.root["content"][1]
        else:       #不需要更新
            return

    def end_turn(self):
        """
        结束回合
        """
        self.send_opt({"type": "finish"})

    def update_view(self, view):
        self.view.nodes.clear()
        for i in self.others:
            i.pos = [-1,-1,-1]
        for node in view:
            temp_node = Node()
            temp_node.pos = node[0]
            temp_node.interprops = node[1]
            if(len(node) >= 3):
                player_index = self.player_id_to_player_index(node[2])
                self.others[player_index].pos = node[0]
                temp_node.player = node[2]
            self.view.nodes.append(temp_node)

    def pos_to_node_index(self, pos):
        for index,node in enumerate(self.view.nodes):
            if node.pos == pos:
                return index
        return -1

    def player_id_to_player_index(self, player_id):
        for index,player in enumerate(self.others):
            if player.id == player_id:
                return index
        return -1

    def get_my_num(self):
        """
        获得自己的编号，返回值为数字
        """
        return self.player.id

    def get_keys(self):
        """
        获得当前拿到的钥匙，返回值为数字列表
        """
        keys = copy.deepcopy(self.player.keys)
        return keys

    def get_check_cd(self):
        """
        获得探查技能的cd，返回值为数字
        """
        return max(self.player.last_use_check + 5 - self.state)
        pass

    def get_neighbors(self, pos):
        """
        获得某节点的相邻点集，pos为节点坐标三元组，返回值为相邻结点组成的列表
        """
        neighbors = []
        for z in range(3):
            for x in range(7):
                for y in range(7):
                    if self.map.elevator_beside(z,x,y,pos[2],pos[0],pos[1]):
                        neighbors.append((x,y,z))

        return neighbors


    def move(self, pos):
        """
        移动，pos为节点坐标三元组，返回值参见通讯格式
        """
        self.send_opt({"type": "action", "action": ["move", pos]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":   #不是本次移动消息的回复，属于局内消息
                self.in_turn()
            else:
                if not self.root["success"]:
                    return copy.deepcopy(self.root)
                else:
                    self.player.hp = self.root["hp"]
                    self.player.pos = pos
                    if self.root["view"]:
                        self.update_view(self.root["view"])
                    return copy.deepcopy(self.root)
                    

    def attack(self, pos, player_id):
        """
        攻击，pos为节点坐标三元组，player_id为攻击目标
        """
        self.send_opt({"type": "action", "action": ["attack", pos, player_id]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":
                self.in_turn()
            else:
                if self.root["success"]:
                    self.others[self.player_id_to_player_index(player_id)].hp -= 40
                return copy.deepcopy(self.root)

    def interact(self, tool_type, capsule=False):
        """
        与场景道具交互，tool_type为道具类型字符串，若tool_type为逃生舱，则capusule代表开始计时或暂停计时
        返回值参见通讯格式
        """
        if tool_type == "EscapeCapsule":
            self.send_opt({"type": "action", "action":["interact", tool_type, capsule]})
        else:
            self.send_opt({"type": "action", "action": ["interact", tool_type]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":
                self.in_turn()
            else:
                return copy.deepcopy(self.root)

    def view_box(self, box_type, tool_type):
        """
        检视箱子，box_type为掉落池/物资点字符串，tool_type为具体道具类型，返回值参见通讯格式
        """
        if box_type == "Materials":
            self.send_opt({"type": "action", "action":["interact", box_type, tool_type]})
        else:
            self.send_opt({"type": "action", "action":["interact", box_type]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":
                self.in_turn()
            else:
                if not self.root["success"]:
                    return copy.deepcopy(self.root)
                else:
                    if tool_type == "key":
                        self.player.keys = self.root["keys"]
                    else:
                        self.player.tools.landmine_number[0] = self.root["tools"]["LandMine"]
                        self.player.tools.sticky_number[0] = self.root["tools"]["Sticky"]
                        self.player.tools.transport = self.root["tools"]["Transport"]
                        self.player.tools.kit = self.root["tools"]["Kit"]
                    return copy.deepcopy(self.root)

    def put_trap(self, trap_type):
        """
        放置陷阱，trap_type为陷阱类型字符串
        """
        self.send_opt({"type": "action", "action": ["trap", trap_type]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":
                self.in_turn()
            else:
                if self.root["success"]:
                    if trap_type == "LandMine":
                        self.player.tools.landmine_number[0] -= 1
                        self.player.tools.landmine_number[1] += 1
                        self.player.tools.landmine_pos.append(self.player.pos)
                    elif trap_type == "Sticky":
                        self.player.tools.sticky_number[0] -= 1
                        self.player.tools.sticky_number[1] += 1
                        self.player.tools.sticky_pos.append(self.player.pos)
                return copy.deepcopy(self.root)

    def use_tool(self, tool_type, transport_pos=None):
        """
        使用道具，tool_type为道具类型字符串
        如果使用瞬移，transport_pos是瞬移目的地
        """
        if tool_type == "Transport":
            self.send_opt({"type": "action", "action": ["tool", tool_type, transport_pos]})
        else:
            self.send_opt({"type":"action", "action": ["tool", tool_type]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":
                self.in_turn()
            else:
                if not self.root["success"]:
                    return copy.deepcopy(self.root)
                else:
                    if tool_type == "Kit":
                        self.player.hp = self.root["hp"]
                    elif tool_type == "Transport":
                        self.player.pos = transport_pos
                        self.player.hp = self.root["hp"]
                        if "view" in self.root:
                            self.update_view(self.root["view"])
                    return copy.deepcopy(self.root)


    def detect(self, detect_pos):
        self.send_opt({"type": "action", "action": ["detect", detect_pos]})
        while True:
            self.receive_data()
            if self.root["type"] != "action":
                self.in_turn()
            else:
                if self.root["success"] == True:
                    self.player.last_use_check = self.state
                return copy.deepcopy(self.root)

    def get_spawn_pos(self,x):
        """
        获得出生点位置坐标，返回值为三元组
        """
        if x == 0:
            return[0,0,1]
        elif x == 1:
            return [6,0,1]
        elif x == 2:
            return [6,6,1]
        elif x == 3:
            return [0,6,1]
        

    def get_escape_pos(self):
        """
        获得撤离点位置坐标，返回值为三元组
        """
        return [3,3,0]

    def get_landmine_pos(self):
        """
        获得当前地雷位置坐标，返回值为三元组列表
        """
        landmine_pos = copy.deepcopy(self.player.tools.landmine_pos)
        return landmine_pos

    def get_sticky_pos(self):
        """
        获得当前粘弹位置坐标，返回值为三元组列表
        """
        sticky_pos = copy.deepcopy(self.player.tools.sticky_pos)
        return sticky_pos

    def get_my_pos(self):
        return copy.deepcopy(self.player.pos)

    def get_other_pos(self,player_id):
        player_index = self.player_id_to_player_index(player_id)
        return copy.deepcopy(self.others[player_index].pos)

    def get_my_hp(self):
        return copy.deepcopy(self.player.hp)
    
    def get_other_hp(self,player_id):
        player_index = self.player_id_to_player_index(player_id)
        return copy.deepcopy(self.others[player_index].hp)

    def get_other_keys(self,player_id):
        player_index = self.player_id_to_player_index(player_id)
        return copy.deepcopy(self.others[player_index].keys)
    # =====================================================================
    # v4 — my own if-else strategy. Game judgement:
    #   Win = collect 4 keys (one per corner KeyMachine) -> reach the
    #   escape capsule at [3,3,0] and start escaping. Keys also drop from
    #   boxes/materials. HP matters (attack = 40 dmg; Kit heals). The zone
    #   shrinks over time, so don't loiter on the edge. 4-player FFA; rank
    #   points 4..1, so even not escaping, surviving + keys beats dying.
    # =====================================================================

    def _node_type(self, x, y, z):
        """Static map type at grid (x,y,z). Map indexes node[z][x][y]."""
        return self.map.node[z][x][y].type

    def _objectives(self):
        """All static objective positions by type: materials(1)/keymachine(2)/escape(4)."""
        mats, machines, escape = [], [], []
        for z in range(3):
            for x in range(7):
                for y in range(7):
                    t = self.map.node[z][x][y].type
                    pos = (x, y, z)
                    if t == 1:
                        mats.append(pos)
                    elif t == 2:
                        machines.append(pos)
                    elif t == 4:
                        escape.append(pos)
        return mats, machines, escape

    def _neighbors_graph(self):
        """All (pos -> [neighbor positions]) once per game (graph is static).

        ``get_neighbors`` scans all 147 cells per call via ``elevator_beside``,
        so a single BFS over ~114 nodes would cost ~16k scans; calling that per
        turn blows the 20s per-action budget. Cache the adjacency up front.
        """
        if hasattr(self, "_nb_cache"):
            return self._nb_cache
        cache = {}
        for z in range(3):
            for x in range(7):
                for y in range(7):
                    pos = (x, y, z)
                    cache[pos] = self.get_neighbors(pos)
        self._nb_cache = cache
        return cache

    def _bfs(self, start):
        """BFS over the move graph from start; returns {pos: (dist, first_step)}."""
        start = tuple(start)
        graph = self._neighbors_graph()
        dist = {start: (0, None)}
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            d, _ = dist[cur]
            for nxt in graph.get(cur, ()):
                if nxt not in dist:
                    first = nxt if d == 0 else dist[cur][1]
                    dist[nxt] = (d + 1, first)
                    queue.append(nxt)
        return dist

    def _step_toward(self, targets, avoid=None):
        """Pick the neighbor that begins the shortest path to any target.

        Returns None if already on a target or nothing reachable. ``avoid`` is
        a set of positions to prefer not stepping onto (enemies/traps); only
        used as a tie-breaker when a clean option exists.
        """
        my_pos = tuple(self.get_my_pos())
        target_set = {tuple(t) for t in targets}
        if my_pos in target_set:
            return None
        dist = self._bfs(my_pos)
        avoid = set(avoid or [])
        best = None          # (distance, first_step) clean
        best_avoided = None  # same, stepping onto an avoided tile
        for t in targets:
            t = tuple(t)
            if t not in dist:
                continue
            d, first = dist[t]
            if first is None:
                continue
            cand = (d, first)
            if first in avoid:
                if best_avoided is None or d < best_avoided[0]:
                    best_avoided = cand
            else:
                if best is None or d < best[0]:
                    best = cand
        chosen = best if best is not None else best_avoided
        return chosen[1] if chosen else None

    def _adjacent_enemy_to_attack(self):
        """Return (enemy_pos, enemy_id) if a lone adjacent enemy is weaker.

        Commit to a fight only when the target is exactly on or next to me, is
        the *only* adjacent enemy (no 2v1), and I have a clear HP edge. Attack =
        40 dmg; two hits kill a fresh (100 hp) player, so press only when ahead.
        """
        my_pos = self.get_my_pos()
        my_hp = self.get_my_hp()
        neighbors = self.get_neighbors(my_pos)
        adj = []
        for pid in range(4):
            if pid == self.get_my_num():
                continue
            ostatus = self.others[self.player_id_to_player_index(pid)].status
            if ostatus in (STATUS.DEAD.value, STATUS.ESCAPED.value):
                continue
            opos = self.get_other_pos(pid)
            if list(opos) == list(my_pos) or tuple(opos) in neighbors:
                adj.append((pid, opos, self.get_other_hp(pid)))
        if len(adj) != 1:
            return None
        pid, opos, ohp = adj[0]
        if my_hp >= ohp + 10:   # clear HP lead -> attack
            return (opos, pid)
        return None

    def _enemy_positions(self):
        ps = []
        for pid in range(4):
            if pid == self.get_my_num():
                continue
            if self.others[self.player_id_to_player_index(pid)].status not in (
                STATUS.DEAD.value, STATUS.ESCAPED.value,
            ):
                ps.append(tuple(self.get_other_pos(pid)))
        return [p for p in ps if p != (-1, -1, -1)]

    def play(self):
        """My if-else decision tree — exactly ONE action per turn, then finish.

        Each branch either performs one action and returns (so start_turn will
        send ``finish``), or falls through to the final move. Chaining a move
        after an interact in the same turn hangs the wire protocol (the logic
        does not echo a second action reply), so we never do it.
        """
        try:
            my_keys = self.get_keys()
            n_keys = len(my_keys)
            my_hp = self.get_my_hp()
            my_pos = tuple(self.get_my_pos())
            _, machines, escape = self._objectives()
            machine_set = {tuple(m) for m in machines}
            escape_set = {tuple(e) for e in escape}

            # Key machines we've already drained (no key returned) -> skip.
            if not hasattr(self, "_drained_machines"):
                self._drained_machines = set()

            # --- 1. ESCAPE: 4 keys -> start the escape at the capsule --------
            if n_keys >= 4 and my_pos in escape_set:
                self.interact("EscapeCapsule", True)
                return

            # --- 2. SURVIVE: heal when low and a Kit is available ------------
            if my_hp <= 130 and self.player.tools.kit > 0:
                self.use_tool("Kit")
                return

            # --- 3. FIGHT: punish a lone weaker adjacent enemy ---------------
            tgt = self._adjacent_enemy_to_attack()
            if tgt is not None:
                opos, pid = tgt
                self.attack(list(opos), pid)
                return

            # --- 4. KEY: standing on an un-drained key machine -> take it ----
            if my_pos in machine_set and my_pos not in self._drained_machines:
                res = self.interact("KeyMachine")
                # A key was granted only if our key count grew this turn.
                if len(self.get_keys()) > n_keys:
                    return
                # No key -> drained; never retry this machine, and don't chain
                # a move this turn (the reply was the turn's one action echo).
                self._drained_machines.add(my_pos)
                return

            # --- 5. LOOT: on a materials node, grab something cheap ----------
            if self._node_type(*my_pos) == 1:  # materials point
                if self.view_box("Box", "Key")["success"]:
                    return
                if self.view_box("Materials", "Kit")["success"]:
                    return
                return  # materials node exhausted this turn; finish here

            # --- 6. NAVIGATE: BFS one step toward the nearest objective ------
            if n_keys >= 4:
                targets = list(escape_set)
            else:
                targets = [m for m in machine_set if m not in self._drained_machines]
                if not targets:
                    mats, _, _ = self._objectives()
                    targets = mats

            step = self._step_toward(targets, avoid=self._enemy_positions())
            if step is not None:
                self.move(list(step))
                return
            # Fallback: any safe neighbor so we never stand still.
            neigh = self.get_neighbors(my_pos)
            enemies = set(self._enemy_positions())
            safe = [n for n in neigh if n not in enemies]
            if safe:
                self.move(list(safe[0]))
        except Exception:
            # Never hang: a crash here just ends the turn.
            pass

    def run(self):
        while True:
            if self.player.status == STATUS.ESCAPED:
                break
            self.receive_data()
            if self.root["type"] == "id":
                self.init_game()
            elif self.root["type"] == "roundbegin":
                self.start_turn()
            elif self.root["type"] == "offround":
                self.off_turn()
            else:
                self.in_turn()

if __name__=="__main__":
    c = AIClient()
    c.run()