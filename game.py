"""
Starseed Protocol - Prototype game using OpenAI (or mock mode when API key missing)

Usage:
  pip install openai~=1.0
  export OPENAI_API_KEY="sk-..."
  python /mnt/data/starseed_protocol.py

This prototype emphasizes the LLM interactions as described in the assignment:
- AI generates planets (name, description, resources)
- Player writes solutions to locally-generated "survival problems"
- AI evaluates player's solution and returns feasibility & asset changes
- Game stores discovered planets and allows transporting resources back to Earth

The OpenAI calls are wrapped so the game still runs without an API key (mock mode).
"""

import json
import os
import random
import time
import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

import prompts

# --- Configuration ---
conf = {}
try:
    conf = json.loads(open("config.json", "r").read())
except Exception:
    raise Exception("Configuration file not found, make sure you put config.json in the same directory.")
try:
    OPENAI_API_KEY = conf['secret-key']
except:
    raise Exception("secret-key not found in config.json.")
MODEL = "gpt-4"  # adjust by your available model
SAVE_PATH = "saves/starseed_save.json"

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

RESOURCE_ADD_INTERVAL = 5 # second(s)

# --- Data classes ---
@dataclass
class Resource:
    name: str
    description: str
    value: int
    transportation_fee: int
    last_transport_time: int = 0

@dataclass
class Planet:
    id: str
    name: str
    description: str
    type: int = 1  # 0: none, 1: resource, 2: enemy
    initial_value: int = 0
    resources: List[Resource] = field(default_factory=list)
    discovered_time: int = 0  # time when discovered

@dataclass
class GameState:
    turn: int = 0
    assets: int = 1000  # starting capital, tweakable
    planets: Dict[str, Planet] = field(default_factory=dict)
    current_planet_id: str = "earth"  # start on Earth
    game_over: bool = False

    def to_json(self):
        def encode(obj):
            if isinstance(obj, Planet) or isinstance(obj, Resource):
                return asdict(obj)
            if isinstance(obj, dict):
                return {k: encode(v) for k, v in obj.items()}
            return obj
        return json.dumps({"turn": self.turn,"assets": self.assets, "planets": encode(self.planets),
                           "current_planet_id": self.current_planet_id, "game_over": self.game_over}, indent=2)

    @staticmethod
    def from_json(json_text):
        raw = json.loads(json_text)
        gs = GameState(turn=raw.get("turn",0), assets=raw.get("assets",1000),
                       current_planet_id=raw.get("current_planet_id","earth"),
                       game_over=raw.get("game_over", False))
        planets_raw = raw.get("planets", {})
        for pid, p in planets_raw.items():
            resources = []
            for r in p.get("resources", []):
                resources.append(Resource(**r))
            planet = Planet(id=pid, name=p.get("name",""), description=p.get("description",""),
                            resources=resources, discovered_time=p.get("discovered_time",0))
            gs.planets[pid] = planet
        return gs

# --- Helpers for OpenAI interaction with fallback mocks ---
def call_llm_for_planet() -> str:
    """
    Calls OpenAI (or other provider) to generate a planet JSON string
    following the schema defined in prompts.NEW_PLANET.
    If API key is missing or API call fails, returns a mocked planet JSON.
    """
    api_key = OPENAI_API_KEY

    if OPENAI_AVAILABLE and api_key:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict JSON output generator."},
                    {"role": "user", "content": prompts.NEW_PLANET}
                ],
                temperature=1.0,
                max_tokens=700
            )
            content = resp.choices[0].message["content"].strip()

            # Ensure output is valid JSON
            json.loads(content)
            return content

        except Exception as e:
            print("OpenAI 不給力啊!", e)

def call_llm_for_problem(prompt: str) -> str:
    """
    Calls OpenAI to generate a problem description based on a planet.
    Falls back to a random problem if API unavailable.
    """
    api_key = OPENAI_API_KEY
    if OPENAI_AVAILABLE and api_key:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一個會產生星球生存問題的 AI。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=300,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            print("OpenAI API error, falling back to mock problem:", e)

    # fallback 預設問題
    fallback = random.choice([
        "突如其來的暴風季摧毀了部分建築，需要立即修復。",
        "食物供應受到未知微生物污染，可能會短缺。",
        "礦脈品質下降，需要重新規劃精煉方式。",
        "敵對生物持續襲擊運輸路線，需要防禦措施。"
    ])
    return fallback


def call_llm_for_evaluation(prompt: str) -> str:
    """
    Calls OpenAI to evaluate a player's proposed solution.
    If no API key, returns a mocked evaluation JSON.
    Expected JSON output keys: {"feasible": bool, "explanation": str, "asset_change": int}
    """
    api_key = OPENAI_API_KEY
    if OPENAI_AVAILABLE and api_key:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role":"system","content":"You are an objective project evaluator returning JSON only."},
                          {"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            print("OpenAI API error, falling back to mock. Error:", e)
    # Mock evaluation: random but somewhat logical
    feasible = random.random() < 0.6
    explanation = "Mock evaluation: " + ("Plan seems feasible given resources." if feasible else "Plan has structural issues (underfunded or risky).")
    asset_change = random.randint( -200, 700) if feasible else random.randint(-500, -50)
    return json.dumps({"feasible": feasible, "explanation": explanation, "asset_change": asset_change}, indent=2)

# --- Game logic functions ---
def generate_planet(game_state: GameState) -> Planet:
    raw = call_llm_for_planet()
    now_time = int(time.time())
    try:
        parsed: dict = json.loads(raw)
    except Exception:
        # Try to extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}")+1
        parsed: dict = json.loads(raw[start:end])
    pid = f"{random.randint(0, 10000)}"
    resources = []
    for r in parsed.get("resources", []):
        r["last_transport_time"] = now_time
        resources.append(Resource(**r))
    i_v = parsed.get("initial_value", 0)
    t = parsed.get("type", 1)
    planet = Planet(id=pid, name=parsed.get("name","Unnamed"), description=parsed.get("description",""),
                    resources=resources, discovered_time=now_time, type=t, initial_value=i_v)
    return planet

def propose_and_evaluate_problem(game_state: GameState, planet: Planet):
    # 使用 AI 生成星球問題
    problem_prompt = prompts.NEW_EVENT + "\n\n星球描述：" + planet.description
    print("正在星球上探索問題... 等待期間來看一下《人工智慧概論》吧！")
    problem = call_llm_for_problem(problem_prompt)

    print(f"\n星球 {planet.name} 遇到的問題（回合 {game_state.turn}）：\n{problem}\n")
    print("請寫下你的解決方案。輸入完成後請按兩次 Enter 提交。\n")

    lines = []
    while True:
        try:
            l = input()
        except KeyboardInterrupt:
            l = ""
        if l.strip() == "":
            break
        lines.append(l)

    solution_text = "\n".join(lines).strip()

    print("正在用你的方法解決問題... 等待期間來看一下《人工智慧概論》吧！")

    # AI 評估你的方案
    prompt = (
        prompts.EVENT_SOLVE
        + "\n\n星球描述：\n" + planet.description
        + "\n\n問題：\n" + problem
        + "\n\n玩家解決方案：\n" + solution_text
    )

    raw_eval = call_llm_for_evaluation(prompt)

    try:
        eval_json = json.loads(raw_eval)
    except Exception:
        s = raw_eval.find("{")
        e = raw_eval.rfind("}") + 1
        eval_json = json.loads(raw_eval[s:e])

    feasible = bool(eval_json.get("feasible", False))
    explanation = eval_json.get("explanation", "")
    asset_change = int(eval_json.get("asset_change", 0))

    print("\n--- AI 評估結果 ---")
    print("可行性:", feasible)
    print("結果說明:", explanation)
    print("資產變化:", asset_change)

    game_state.assets += asset_change

    return {
        "feasible": feasible,
        "explanation": explanation,
        "asset_change": asset_change
    }

def explore_new_planet(game_state: GameState):
    cost = 2000 + int(((len(game_state.planets) - 1)**1.7) * 350)
    print(f"這次探索任務花費: ${cost} .")
    if game_state.assets < cost:
        print("我們需要錢錢...")
        return None
    game_state.assets -= cost
    print("正在從茫茫星海中探索新星球... 等待期間來看一下《人工智慧概論》吧！")
    planet = generate_planet(game_state)
    if planet is None:
        game_state.assets += cost
        print("出了一點小意外，本次旅程失敗，花費會全額退還")
        return None
    game_state.planets[planet.id] = planet
    print(f"發現星球 {planet.name} (id {planet.id})\n說明: {planet.description}\n")
    if planet.type == 0:
        print("這裡什麼都沒有...")
    elif planet.type == 1:
        print("這裡有資源！")
        for r in planet.resources:
            print(f" - {r.name}: 價值={r.value}, 運輸費用={r.transportation_fee}")
        print(f"回程時順便帶了一些戰利品，獲得 ${planet.initial_value}")
        game_state.assets += planet.initial_value
    elif planet.type == 2:
        print("完蛋，遭到敵人入侵！")
        print(f"與他們戰鬥又損失了 ${-planet.initial_value}")
        game_state.assets += planet.initial_value
    return planet

def transport_resources(game_state: GameState, planet: Planet):
    now_time = int(time.time())
    print(f"從 {planet.name} 運輸資源到 Earth. 目前資產: ${game_state.assets}")
    for i, r in enumerate(planet.resources):
        now_value = min((now_time - r.last_transport_time), 180) // 15 * r.value - r.transportation_fee
        ltt = datetime.datetime.fromtimestamp(r.last_transport_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i}. {r.name} | 價值={r.value} | 運費={r.transportation_fee} | 扣除運費淨利潤={now_value}| 上次運輸時間={ltt}")
    print("輸入要輸送資源的項次 (直接按 Enter 取消操作):")
    sel = input().strip()
    if sel == "":
        return
    try:
        idx = int(sel)
        r = planet.resources[idx]
    except Exception:
        print("Invalid selection.")
        return
    # compute gain: value * (turn - last_transport_time) - fee
    gain = min((now_time - r.last_transport_time), 180) // 15 * r.value - r.transportation_fee
    if game_state.assets < r.transportation_fee:
        print("哇，連運費都付不起了 : (")
        return
    # pay transportation fee then add value*delta_time
    game_state.assets += gain
    r.last_transport_time = now_time
    print(f"成功運送 {r.name}. 資產變化: ${ gain }")
    print("目前資產: $", game_state.assets, sep="")

def save_game(game_state: GameState, path=SAVE_PATH):
    with open(path, "w") as f:
        f.write(game_state.to_json())
    print("存檔已存放到: ", path)

def load_game(path=SAVE_PATH) -> Optional[GameState]:
    if not os.path.exists(path):
        print("未找到在以下路徑的存檔: ", path)
        return None
    with open(path, "r") as f:
        txt = f.read()
    gs = GameState.from_json(txt)
    print("已載入存檔. 回合:", gs.turn, f"資產: ${gs.assets}", "探索星球數量:", len(gs.planets))
    return gs

# --- Initialize default Earth as starting planet ---
def initial_game_state() -> GameState:
    now_time = int(time.time())
    gs = GameState(turn=0, assets=1500, current_planet_id="0", game_over=False)
    earth = Planet(id="0", name="Earth", description="起始星球: 地球", discovered_time=now_time)
    # Earth has modest resources to begin
    earth.resources = [
        Resource(name="木材", description="可以提供建築等材料.", value=10, transportation_fee=5, last_transport_time=now_time),
        Resource(name="金屬礦石", description="基本金屬礦石.", value=25, transportation_fee=15, last_transport_time=now_time),
    ]
    gs.planets[earth.id] = earth
    return gs

# --- Main CLI loop ---
def main_loop():
    print("=== Starseed Protocol (Prototype) ===\n")
    gs = load_game() or initial_game_state()
    while not gs.game_over:
        print("\n--- Control Panel ---")
        print(f"資產: ${gs.assets}")
        print("目前探索到的星球:")
        for pid, p in gs.planets.items():
            dtt = datetime.datetime.fromtimestamp(p.discovered_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f" - {p.name} (id={pid}) 資源={len(p.resources)} 發現時的時間={dtt}")
        print("\n可執行的動作：")
        print(" 1) 處理星球上的在地問題（從清單中選擇星球）")
        print(" 2) 探索新星球（花費會隨回合增加）")
        print(" 3) 將資源運送回地球")
        print(" 4) 儲存遊戲")
        print(" 5) 載入遊戲")
        print(" 6) 離開遊戲")

        cmd = input("請選擇動作編號：").strip()

        if cmd == "1":
            # choose planet
            print("請輸入欲處理問題的星球 ID：")
            pid = input().strip()
            if pid in gs.planets:
                propose_and_evaluate_problem(gs, gs.planets[pid])
            else:
                print("星球 ID 不存在。")

        elif cmd == "2":
            explore_new_planet(gs)

        elif cmd == "3":
            print("要從哪個星球運送資源(填入 id)？")
            pid = input().strip()
            if pid in gs.planets:
                transport_resources(gs, gs.planets[pid])
            else:
                print("星球 ID 不存在。")

        elif cmd == "4":
            save_game(gs)

        elif cmd == "5":
            loaded = load_game()
            if loaded:
                gs = loaded

        elif cmd == "6":
            print("確定要離開？需要先儲存遊戲嗎？ (y/n)")
            if input().lower().startswith("y"):
                save_game(gs)
            break

        else:
            print("無效的指令。")

        # quick bankruptcy check
        if gs.assets < -1000:
            print("負債過高，已無法挽回。遊戲結束。")
            gs.game_over = True

        print(f"目前資產: ${gs.assets}")

    print("感謝遊玩《Starseed Protocol 星辰種子計畫》！你的最終資產為：", gs.assets)


if __name__ == "__main__":
    main_loop()