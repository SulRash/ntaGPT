from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from tqdm import tqdm
import random

seed = 42

random.seed(seed)

# Map-to-Obstacle for generating path.
MTO = {
    "S": 1,
    "G": 1,
    "F": 1,
    "H": 0
}

def map_to_obstacle(char: str):
    return MTO[char]

# Map-to-Token to avoid the tokenization step.
# 0, 1, and 2 are reserved for <eor>, <eom>, and \n respectively
MTT = {
    "S": "3",
    "G": "4",
    "F": "5",
    "H": "6"
}

TTM = {
    0: "<eor>",
    1: "<eom>",
    2: "\n",
    3: "S",
    4: "G", 
    5: "F",
    6: "H"
}

def tokenizer(string: str):
    string = string.replace("<eor>", "0")
    string = string.replace("<eom>", "1")
    string = string.replace("\n", "2")
    string = string.replace("S", MTT["S"])
    string = string.replace("G", MTT["G"])
    string = string.replace("F", MTT["F"])
    string = string.replace("H", MTT["H"])
    return list(map(int, list(string)))

def detokenizer(ids: List[int]):
    return ''.join([TTM[i] for i in ids])

def generate_map(map_size: int = 8, pt: bool = False) -> List[List[str]]:
    random_map = generate_random_map(size=map_size, seed=seed)

    # If pretraining, randomize player state
    if pt:
        player = random.randint(0,(map_size*2)-1)
        random_map = list(''.join(random_map))
        random_map[0] = "F"
        random_map[player] = "S"
        random_map = ''.join(random_map)
        random_map = [random_map[x:x+8] for x in range(0,len(random_map),8)]
    return random_map

def generate_path(game_map: List[List[str]]) -> Tuple[List[int],int]:
    game_map = [list(map(map_to_obstacle, x)) for x in game_map]
    
    grid = Grid(matrix=game_map)

    start = grid.node(0,0)
    corner = len(game_map) - 1
    end = grid.node(corner, corner)

    finder = AStarFinder()
    path, _ = finder.find_path(start, end, grid)
    return path

def map_to_str(game_map: List[List[str]]) -> str:
    map_entry = ""
    for row in game_map:
        map_entry += "".join(row) + "<eor>\n"
    map_entry += "<eom>\n\n"
    return map_entry

def generate_im_pair(game_map: List[List[str]], path: List[Grid]) -> Dict[str, str]:
    previous_move = [0,0]
    pair = {"init": map_to_str(game_map), "moves": ""}
    
    for move in path[1:]:
        moves = [move.x, move.y]
        diff = [a - b for a, b in zip(moves, previous_move)]
        row = list(game_map[move.x])
        row[move.y] = "S" 
        game_map[move.x] = "".join(row)
        row = list(game_map[move.x - diff[0]])
        row[move.y - diff[1]] = "F"
        game_map[move.x - diff[0]] = "".join(row)

        previous_move = moves

        pair["moves"] += map_to_str(game_map)
    return pair

def generate_pt_data_mp(args):
    maps, map_size, temp_dir, save_as, process_id = args
    game_maps = []
    for _ in tqdm(range(maps), desc=f"Process {process_id}", position=process_id):
        game_map = generate_map(map_size=map_size, pt=True)
        game_maps.append(map_to_str(game_map))
    
    if save_as == "txt":
        with open(f"{temp_dir}/pt_data_process_{process_id}.txt", "w") as file:
            for game_map in game_maps:
                file.write(game_map)

def generate_pt_data(maps: int = 10000000, map_size: int = 8, output_dir: str = "generated_data/pt_data", save_as: str = "txt", num_processes: int = 4) -> None:
    chunk_size = maps // num_processes
    temp_dir = "temp_pt_data"
    os.makedirs(temp_dir, exist_ok=True)

    args = [(chunk_size, map_size, temp_dir, save_as, i) for i in range(num_processes)]
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_pt_data_mp, args), total=num_processes, desc="Generating Pretraining Data"))

    # Concatenate temporary files into the final pt_data file
    with open(f"{output_dir}.txt", "w") as output_file:
        for i in range(num_processes):
            temp_file = f"{temp_dir}/pt_data_process_{i}.txt"
            with open(temp_file, "r") as file:
                output_file.write(file.read())
            os.remove(temp_file)
    os.rmdir(temp_dir)

def generate_ft_data_mp(args):
    maps, map_size, temp_dir, save_as, process_id = args
    data = {"train": []}

    for _ in tqdm(range(maps), desc=f"Process {process_id}", position=process_id):
        game_map = generate_map(map_size=map_size, pt=False)
        path = generate_path(game_map)
        pair = generate_im_pair(game_map, path)
        data["train"].append(pair)

    if save_as == "txt":
        with open(f"{temp_dir}/ft_data_process_{process_id}.txt", "w") as file:
            for game_map in data["train"]:
                file.write(game_map["init"] + game_map["moves"])

def generate_ft_data(maps: int = 250000, map_size: int = 8, output_dir: str = "generated_data/ft_data", save_as: str = "txt", num_processes: int = 4) -> None:
    chunk_size = maps // num_processes
    temp_dir = "temp_ft_data"
    os.makedirs(temp_dir, exist_ok=True)

    args = [(chunk_size, map_size, temp_dir, save_as, i) for i in range(num_processes)]
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_ft_data_mp, args), total=num_processes, desc="Generating Finetuning Data"))

    # Concatenate temporary files into the final ft_data file
    with open(f"{output_dir}.txt", "w") as output_file:
        for i in range(num_processes):
            temp_file = f"{temp_dir}/ft_data_process_{i}.txt"
            with open(temp_file, "r") as file:
                output_file.write(file.read())
            os.remove(temp_file)
    os.rmdir(temp_dir)
    
if __name__ == "__main__":
    import os
    import yaml
    import sys

    pt_maps = None
    ft_maps = None
    map_size = None

    with open(sys.argv[1]) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    globals().update(config)

    os.makedirs("generated_data", exist_ok=True)
    generate_pt_data(maps=pt_maps, map_size=map_size, num_processes=cpu_count())
    generate_ft_data(maps=ft_maps, map_size=map_size, num_processes=cpu_count())