import json, language_tool_python, os, pygame, random, torch
import tkinter as tk
from PIL import Image, ImageTk
from transformers import pipeline
from tkinter import scrolledtext, Label, Listbox, Frame, Button


print(torch.__version__)

# AI Model for drone conversations
if torch.cuda.is_available():
    ai_chatbot = pipeline("text-generation", model="gpt2", device=0)
else:
    ai_chatbot = pipeline("text-generation", model="gpt2")


# Game data storage
DATA_FILE = "player_responses.json"

def load_responses():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}


def ensure_complete_sentence(text):
    if text and text[-1] not in ".!?":
        return text.rstrip(",;") + "."
    return text


def correct_text(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text


# Unique ID for tracking responses
player_id = "Player"

def save_response(question, answer):
    completed_answer = ensure_complete_sentence(answer)
    corrected_answer = correct_text(completed_answer).strip()

    # Silently ignore responses that are too short
    if len(corrected_answer) < 10:
        return

    data = load_responses()

    if player_id not in data:
        data[player_id] = []

    data[player_id].append({"question": question, "answer": completed_answer})

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def get_past_response(question):
    data = load_responses()
    matching_answers = []  # List to store previous responses

    for responses in data.values():
        for entry in responses:
            if entry["question"].strip().lower() == question.strip().lower():
                matching_answers.append(entry["answer"])

    if matching_answers:
        return random.choice(matching_answers)

    return "I don't know."  # Default fallback if no matching response exists


# Game world structure
game_map = {
    # Research Areas
    "AI_Core": {
        "desc": "A dimly lit chamber, illuminated only by flickering holographic displays. In the center of the room, "
                "a monolithic AI server hums ominously, its circuits pulsating with orange light. Scattered across the "
                "floor are half-disassembled androids, their hollow eyes staring blankly. The air crackles with the faint "
                "whisper of machine learning models running endless calculations. It feels like you are being watched.",
        "alt_desc": "This room has glowing lights and a big computer. It feels a little spooky, like something is watching.",
        "image": "ai_core.png",
        "exits": {"south": "AI_Lab"},
        "items": []
    },
    "Storage_Room": {
        "desc": "This cramped, dust-ridden storage room smells of stale air and old electronics. Metal shelves line the walls, "
                "filled with outdated AI research notes and stacks of old hard drives. A single fluorescent light flickers, "
                "casting eerie shadows over the scattered equipment. Rusted tools and disassembled robotic arms are piled in the "
                "corner, relics of forgotten experiments.",
        "alt_desc": "A small dusty room filled with shelves and boxes. You see lots of old stuff and wires.",
        "image": "storage_room.png",
        "exits": {"south": "Data_Center"},
        "items": []
    },
    "Cybernetics_Lab": {
        "desc": "The hum of machinery fills the air as robotic arms twitch and flicker with electricity. Transparent tanks "
                "filled with strange neural gel line the room, each containing incomplete cybernetic bodies, their synthetic "
                "skin peeled back to reveal underlying circuits. A dim red light on the far terminal blinks slowly, as if waiting "
                "for input. Diagrams of humanoid augmentation cover the walls, detailing experiments that blur the line between "
                "man and machine.",
        "alt_desc": "Machines and parts are everywhere. Some of them look like robots being built or fixed.",
        "image": "cybernetics_lab.png",
        "exits": {"south": "Testing_Chamber"},
        "items": []
    },
    "AI_Lab": {
        "desc": "This high-tech research lab is packed with machinery designed for AI experimentation. Large server racks hum "
                "along the walls, their cables running like veins through the facility. On the far end, a transparent console "
                "flashes data from past simulations. Several monitors line the walls leading up to it. Despite the "
                "advanced technology surrounding you, the room feels oddly abandoned, as if something was left unfinished.",
        "alt_desc": "A clean lab filled with blinking machines. Screens show graphs and numbers, but no one is here.",
        "image": "lab.png",
        "exits": {"north": "AI_Core", "east": "Data_Center", "south": "Security_Office"},
        "items": []
    },
    "Data_Center": {
        "desc": "This cavernous data center is lined with towering server stacks, their blinking LEDs casting an eerie glow. "
                "The hum of cooling fans creates a low, almost meditative background noise. Cables snake across the floor, "
                "forming an artificial jungle of technology. In the center, a control panel is powered down, its screen dark. "
                "A faint beep echoes through the room at irregular intervals, like a heartbeat of the system that never sleeps.",
        "alt_desc": "A big room with lots of computer towers and blinking lights. It hums quietly.",
        "image": "data_center.png",
        "exits": {"west": "AI_Lab", "north": "Storage_Room", "east": "Testing_Chamber", "south": "Break_Room"},
        "items": []
    },
    "Testing_Chamber": {
        "desc": "Large mechanical arms hang from the ceiling, their rusted joints barely moving as they spark with electricity. "
                "An inactive AI-controlled drone sits in the center of the chamber, covered in a thick layer of dust. The air is "
                "heavy with the scent of burnt circuits. Safety glass separates you from the drone inside the chamber. "
                "You feel a strange unease standing here, as if the machines are waiting for something.",
        "alt_desc": "Robotic arms hang from the ceiling. A dusty drone sits quietly in the middle.",
        "image": "testing_chamber.png",
        "exits": {"west": "Data_Center", "north": "Cybernetics_Lab", "south": "Power_Generator"},
        "items": []
    },

    # 🏢 Office Areas
    "Security_Office": {
        "desc": "The scent of stale coffee lingers in this security monitoring room. Rows of monitors display security feeds, some static-filled, "
                "others showing empty hallways. A keypad on the far wall locks a cabinet. A sticky note with faded ink is stuck next to it. "
                "A desk with an old computer is covered in logs of security breaches, most of them flagged but unresolved.",
        "alt_desc": "A small office with screens showing different rooms. There’s a locked cabinet on the wall.",
        "image": "security_office.png",
        "exits": {"north": "AI_Lab", "east": "Break_Room"},
        "locked": True,
        "required_item": "Keycard",
        "items": ["USB"]
    },
    "Break_Room": {
        "desc": "A quiet break room with soft, flickering ceiling lights and a row of empty chairs surrounding a long metal table. "
                "A vending machine stands near the wall, its glass cracked but still stocked with old snacks, next to a water cooler that hums gently. "
                "A faded evacuation plan is pinned up — someone has marked over it in red ink with the words: 'DON’T TRUST THEM.' "
                "Scribbled all over the wall is a map of many numbers connected together by lines, in some sort of nonsensical pattern.",
        "alt_desc": "A messy room with a broken vending machine and water cooler. Someone scribbled numbers on the wall to form some sort of pattern...",
        "image": "break_room.png",
        "exits": {"west": "Security_Office", "north": "Data_Center", "south": "Executive_Office"},
        "items": []
    },
    "Rooftop_Observation": {
        "desc": "Perched atop the facility, the rooftop observation deck offers a sweeping view of the desolate landscape. " 
             "The wind howls through the skeletal remains of broken railings, carrying the distant echoes of something unseen. "
             "A powerful telescope stands bolted to the platform, its cracked lens still pointed at a cluster of stars. "
             "Scattered across the floor are old research notes, some half-buried under layers of dust, their ink smudged "
             "by years of exposure. The sky above is vast and unnervingly silent, save for the occasional flicker of a failing satellite. "
             "You feel both infinitely small and eerily watched.",
        "alt_desc": "You're on the roof! There's a big telescope and a great view of the sky.",
        "image": "rooftop.png",
        "exits": {"east": "Executive_Office"},
        "items": []
    },
    "Executive_Office": {
        "desc": "Once the nerve center of the facility, this office now feels like a tomb of forgotten decisions. "
             "A sleek, black desk dominates the center of the room, its surface littered with old data slates flickering with corrupted files. "
             "Holographic displays hover in mid-air, replaying snippets of corporate meetings that cut off mid-sentence. "
             "The walls are lined with shelves of archived reports, many of which have been pulled to the ground in disarray. ",
        "alt_desc": "A fancy office with floating screens and shelves full of papers. It feels quiet.",
        "image": "executive_office.png",
        "exits": {"west": "Rooftop_Observation", "north": "Break_Room", "south": "Abandoned_Parking_Lot"},
        "items": []
    },

    # Outdoor Areas
    "Facility_Courtyard": {
        "desc": "Once a tranquil sanctuary for workers, the courtyard is now a wild expanse of overgrown grass and cracked stone paths. "
             "The remains of old benches are scattered around, their wooden frames warped and rotting. "
             "A dried-up fountain stands in the center, filled with rusted coins—wishes long forgotten. "
             "Vines creep up the sides of the surrounding buildings, nature reclaiming the abandoned space. ",
        "alt_desc": "You're outside! There are old benches and a dry fountain. Plants are growing everywhere.",
        "image": "facility_courtyard.png",
        "exits": {"east": "Abandoned_Parking_Lot"},
        "items": ["Star Map"]
    },
    "Abandoned_Parking_Lot": {
        "desc": "Once a bustling parking lot filled with workers' vehicles, this place is now a wasteland of cracked pavement "
             "and invasive weeds pushing through the asphalt. Rusted, skeletal remains of cars stand frozen in place, their windshields shattered "
             "and their tires long deflated. Something about this place feels eerily abandoned—like whoever left here never meant to leave at all.",
        "alt_desc": "An empty parking lot with old, broken cars. Grass is growing through the ground.",
        "image": "parking_lot.png",
        "exits": {"west": "Facility_Courtyard", "north": "Executive_Office", "east": "AI_Graveyard", "south": "Deserted_Highway"},
        "items": []
    },
    "AI_Graveyard": {
        "desc": "A scrapyard filled with the remains of discarded AI husks. Shattered exoskeletons and broken android limbs are buried in the sand. "
                "Near the center of the yard, a pile of dirt looks suspiciously loose, as if something was buried there recently.",
        "alt_desc": "A sandy place with robot parts buried here and there. Something might be hidden.",
        "image": "ai_graveyard.png",
        "exits": {"west": "Abandoned_Parking_Lot", "north": "Maintenance_Tunnels", "south": "Forest_Outskirts"},
        "items": ["AI Disruptor"]
    },
    "Deserted_Highway": {
        "desc": "The cracked remains of a once-thriving highway stretch endlessly in both directions, disappearing into the misty horizon. "
             "Old streetlights line one side of the road, their bulbs shattered, leaving only rusted husks. "
             "Occasional wrecks of vehicles lie abandoned along the shoulder, doors ajar, their contents looted or long-decayed. "
             "The silence is deafening, broken only by the occasional gust of wind carrying distant echoes of something that no longer exists. "
             "The air here feels different, heavier, as if the highway itself is waiting for something—or someone—to pass through.",
        "alt_desc": "An empty road with old signs and broken lights. Cars are left behind.",
        "image": "highway.png",
        "exits": {"north": "Abandoned_Parking_Lot", "east": "Forest_Outskirts", "south": "Flooded_Reactor"},
        "items": []
    },
    "Forest_Outskirts": {
        "desc": "Thick trees loom over you, their twisted roots snaking through the soft dirt. The air smells fresh but carries an eerie stillness. "
                "Leaves rustle gently, though there is no wind. The ground here is uneven, dotted with half-buried machinery and overgrown pathways.",
        "alt_desc": "You're near a forest. Trees and plants are everywhere, and it's very quiet.",
        "image": "forest.png",
        "exits": {"west": "Deserted_Highway", "north": "AI_Graveyard"},
        "items": ["Rope"]
    },

    # Underground Areas
    "Power_Generator": {
        "desc": "A massive turbine dominates the room, its metallic blades corroded with rust. A maintenance panel flickers with warning messages. "
                "Overhead, thick cables pulse with energy, lighting up the dimly lit chamber in eerie flashes. Something rumbles in the pipes, "
                "an unsettling reminder that not everything here is dormant.",
        "alt_desc": "A big machine fills the room. Some wires are glowing and buzzing quietly.",
        "image": "power_generator.png",
        "exits": {"north": "Testing_Chamber", "south": "Maintenance_Tunnels"},
        "items": []
    },
    "Maintenance_Tunnels": {
        "desc": "Dim, flickering emergency lights cast long, wavering shadows along the narrow, damp tunnel walls. "
                "The air here is thick with the scent of rust and mildew, mingling with the faint chemical tang of leaking coolant. "
                "Pipes stretch along the ceiling like metallic veins, some rattling occasionally, as if something unseen moves within them. "
                "Broken maintenance panels lay scattered across the floor, alongside forgotten tools coated in grime. "
                "The walls are lined with warning signs, their lettering faded beyond recognition. "
                "Every sound you make echoes for far too long, creating the unsettling impression that someone—or something—might be listening.",
        "alt_desc": "Dark tunnels with flickering lights and old pipes. You hear strange echoes.",
        "image": "maintenance_tunnels.png",
        "exits": {"north": "Power_Generator", "south": "AI_Graveyard"},
        "items": []
    },
    "Underground_Research_Lab": {
        "desc": "The underground research lab is in complete disarray. Cryogenic pods line the walls, their interiors fogged over. "
                "Scattered files and broken equipment indicate an abrupt evacuation. A heavy steel door stands at the far end, "
                "sealed with a biometric lock that flashes red when approached. The air is unnervingly cold, and faint whispers echo "
                "through the abandoned facility... ",
        "alt_desc": "This lab is empty and cold. There are strange machines and foggy glass tubes.",
        "image": "underground_lab.png",
        "exits": {"east": "Flooded_Reactor"},
        "locked": True,
        "required_item": "Lab Key",
        "items": ["Battery"]
    },
    "Flooded_Reactor": {
        "desc": "The reactor chamber is partially submerged, with dark, uninviting water lapping at the rusted metal walkways. "
                "A dim, pulsing light emanates from the depths of the flooded section, hinting at something metallic resting just "
                "beneath the surface. Broken cables dangle from the ceiling, sparking occasionally. The smell of ozone lingers heavily in the air, "
                "mixing with the scent of stagnant water.",
        "alt_desc": "Water covers the floor. A shiny object is stuck in the metal below.",
        "image": "reactor.png",
        "exits": {"west": "Underground_Research_Lab", "north": "Deserted_Highway"},
        "items": ["Crowbar"]
    }
}

# Dictionary containing additional puzzle descriptions
puzzle_descriptions = {
    "Keycard": "At one end of the room, you notice an old computer terminal displaying an encrypted security log. "
                "It looks like you'll need to recall a sequence of numbers to decrypt it and retrieve a Keycard.",

    "USB": "A locked cabinet sits against the wall, its access panel blinking with a numerical keypad. "
            "It looks like it requires a 3-digit combination to open, and you suspect something valuable is inside.",

    "Star Map": "An ancient star chart is embedded into a wall display, its symbols worn but still decipherable. "
                 "It appears to be a logic puzzle, where only those with knowledge of planetary alignment can retrieve its secrets.",

    "Lab Key": "An android patrolling the area challenges you with a riddle that it claims only a human would be able to solve. "
                "It claims to have a key to a hidden research lab, but it will only part with it if you answer correctly.",
}
alt_puzzle_descriptions = {
    "Keycard": "There's a terminal you can log into, but it needs a password of numbers...",
    "USB": "A locked cabinet needs a 3-digit code to open. Maybe you can crack it!",
    "Star Map": "A cool puzzle shows planets and stars. You need to figure something out using it.",
    "Lab Key": "A friendly android asks you a riddle. If you answer right, you get a key!",
}
keycard_code = "".join(str(random.randint(0, 9)) for _ in range(8))

special_item_descriptions = {
    "Rope": "A length of rope hangs from a tree branch. It seems sturdy.",
    "Crowbar": "You see a crowbar wedged between two rusted pipes. It looks like it could be retrieved with a sturdy rope.",
    "Wrench": "You notice a vent in the wall. Deep inside, you can barely see a metal object. It looks like a wrench, but you'll need something to pry open the vent.",
    "Battery": "A broken robot is lying on the ground, its fist clenched very tightly around a battery.",
    "Shovel": "Partially buried under a pile of rubble, a rusted shovel catches your eye. It might come in handy for digging."
}

item_tooltips = {
    "Rope": "A sturdy length of rope. Useful for retrieving items from deep or hard-to-reach places.",
    "Crowbar": "A heavy tool capable of prying open vents or containers.",
    "Wrench": "A mechanical tool, perfect for fixing broken equipment.",
    "Battery": "Provides energy to power devices or recharge tools.",
    "Shovel": "Can be used to dig up buried items.",
    "USB": "Contains encrypted data. May unlock some information when used in the right place.",
    "Keycard": "Grants access to locked areas secured by electronic locks.",
    "Star Map": "A chart of constellations. Might help you solve astronomical puzzles.",
    "Lab Key": "Unlocks the underground research lab.",
    "AI Disruptor": "A mysterious weapon capable of disabling AI drones. Must be charged first.",
    "AI Disruptor (Charged)": "Fully charged. Can disable AI drones. Use with caution!"
}


# Predefined locked rooms
locked_rooms = {
    "Security_Office": "Keycard",
    "Underground_Research_Lab": "Lab Key"
}

# Items and their possible locations
item_locations = {
    "Keycard": ["Storage_Room", "Executive_Office"],
    "Lab Key": ["Data_Center", "Testing_Chamber"],
    "Wrench": ["AI_Lab", "Cybernetics_Lab"],
    "Shovel": ["Maintenance_Tunnels", "Deserted_Highway"]
}
assigned_items = {}

for item, locations in item_locations.items():
    while True:
        assigned_room = random.choice(locations)
        if assigned_room not in locked_rooms:
            break

    assigned_items[item] = assigned_room

    if "items" not in game_map[assigned_room]:
        game_map[assigned_room]["items"] = []

    assigned_items[item] = assigned_room
    game_map[assigned_room]["items"].append(item)

# Possible PEMDAS math problems for USB puzzle
USB_math_problems = {
    "What is 121 + 300 / 2?": "271",   # Division first, then addition
    "What is 452 - 150 * 2?": "152",   # Multiplication before subtraction
    "What is 803 - 240 / 3?": "723",   # Division before subtraction
    "What is 554 - 180 / 6?": "524",   # Division before subtraction
    "What is 995 - 330 / 3?": "885",   # Division before subtraction
    "What is 1006 - 450 * 2?": "106"   # Multiplication before subtraction
}
USB_problem, USB_solution = random.choice(list(USB_math_problems.items()))

# Possible questions for the Star Map puzzle
star_map_questions = {
    "Which planet is the closest to the Sun?": "Mercury",
    "Which planet is the second from the Sun?": "Venus",
    "Which planet is the third from the Sun?": "Earth",
    "Which planet is the fourth from the Sun?": "Mars",
    "Which planet is the fifth from the Sun?": "Jupiter",
    "Which planet is the sixth from the Sun?": "Saturn",
    "Which planet is the seventh from the Sun?": "Uranus",
    "Which planet is the furthest from the Sun?": "Neptune",
}
star_map_question, star_map_answer = random.choice(list(star_map_questions.items()))

riddles = {
    "What has keys but can't open locks?": "piano",
    "The more you take, the more you leave behind. What am I?": "footsteps",
    "I speak without a mouth and hear without ears. What am I?": "echo"
}
riddle, lab_key_answer = random.choice(list(riddles.items()))

# List of 15 drones (10 AI and 5 Humans)
drone_model_numbers = [
    "X-100", "Z-205", "M-330", "A-411", "K-512",
    "L-608", "O-702", "G-809", "S-910", "B-1011",
    "T-1112", "C-1213", "H-1314", "R-1415", "V-1516"
]

# Human names (assigned randomly, but hidden until the end)
human_names = [
    "Dr. Carter", "Sarah Mills", "Agent Novak", "Kyle Tran", "Sophia Reed",
    "Liam Mercer", "Evelyn Brooks", "Noah Patel", "Isabella Chen", "Lucas Ford",
    "Zachary Hayes", "Olivia Monroe", "Benjamin Carter", "Emily Dawson", "Nathan Sullivan"
]

# Randomly shuffle the human and AI statuses
random.shuffle(drone_model_numbers)
random.shuffle(human_names)

# Assign 5 drones as humans, 10 as AI
drone_roles = {drone_model_numbers[i]: "Human" for i in range(5)}
drone_roles.update({drone_model_numbers[i]: "AI" for i in range(5, 15)})
drone_real_names = {drone_model_numbers[i]: human_names[i] for i in range(15)}

# List of rooms where drones can appear
drone_rooms = [
    "Storage_Room", "Cybernetics_Lab", "AI_Lab",
    "Security_Office", "Break_Room", "Power_Generator",
    "Rooftop_Observation", "Executive_Office", "Maintenance_Tunnels",
    "Facility_Courtyard", "Abandoned_Parking_Lot", "Deserted_Highway",
    "AI_Graveyard", "Underground_Research_Lab", "Flooded_Reactor"
]
random.shuffle(drone_rooms)

# Dictionary to store drones and their assigned locations
drone_locations = {drone_model_numbers[i]: drone_rooms[i] for i in range(15)}

# List of 15 possible questions drones might ask
drone_questions = [
    "What do you think about AI ethics?",
    "How do you define intelligence?",
    "Do you believe machines can have emotions?",
    "What separates humans from AI?",
    "Is it ethical to replace human workers with AI?",
    "What role should AI play in education?",
    "Do you think AI should be given rights?",
    "Can AI ever be truly creative?",
    "What scares you most about artificial intelligence?",
    "Would you trust an AI doctor?",
    "What do you think about self-driving cars?",
    "Should AI-generated art be considered real art?",
    "How do you feel about AI in military applications?",
    "Can AI understand human emotions?",
    "Do you think AI will ever surpass humans?"
]
random.shuffle(drone_questions)  # Shuffle to distribute among drones

# Store which question each drone will ask
drone_assigned_questions = {drone: drone_questions[i % len(drone_questions)] for i, drone in enumerate(drone_model_numbers)}

# Player State
player_location = "Abandoned_Parking_Lot"
known_drones = set()
inventory = []


# Tooltip class for hover descriptions
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None

        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x = y = 0
        x += self.widget.winfo_rootx() + 50
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, bg="black", fg="white",
                         relief="solid", borderwidth=1, font=("Arial", 10, "normal"))
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


mature_mode = False

def prompt_mode_selection():
    def set_mode_and_start(mode):
        global mature_mode
        mature_mode = (mode == "Mature")
        mode_window.destroy()

    mode_window = tk.Tk()
    mode_window.title("Select Game Mode")
    mode_window.geometry("400x200")
    mode_window.configure(bg="#444444")

    label = Label(mode_window, text="Choose your game mode:", bg="#444444", fg="white", font=("Arial", 14))
    label.pack(pady=20)

    young_button = Button(mode_window, text="Young (Under 13)", width=20, command=lambda: set_mode_and_start("Young"))
    mature_button = Button(mode_window, text="Mature (13+)", width=20, command=lambda: set_mode_and_start("Mature"))

    young_button.pack(pady=10)
    mature_button.pack(pady=10)

    mode_window.mainloop()

prompt_mode_selection()


# Music setup
pygame.mixer.init()
pygame.mixer.music.load("ambient_loop.wav")
pygame.mixer.music.set_volume(0.025)          # Volume between 0.0 and 1.0
pygame.mixer.music.play(-1)                  # -1 loops infinitely

# GUI Setup
root = tk.Tk()
root.title("AI Adventure Game")
root.state("zoomed")
root.configure(bg="#555555")  # Dark gray background

# Configure the grid layout
for i in range(12):
    root.columnconfigure(i, weight=1)
for i in range(6):
    root.rowconfigure(i, weight=1)

# Chat Log
chat_log = scrolledtext.ScrolledText(
    root, width=60, height=25,
    state=tk.NORMAL, bg="#999999", fg="black", wrap="word",
    exportselection=0
)
chat_log.config(state=tk.DISABLED)
chat_log.grid(row=0, column=0, rowspan=4, columnspan=4, padx=5, pady=5, sticky="nsew")

# Drone List
drone_frame = Frame(root, bg="#555555")
drone_frame.grid(row=0, column=10, rowspan=4, columnspan=2, padx=5, pady=5, sticky="nsew")
drone_label = Label(drone_frame, text="Discovered Drones", bg="#555555")
drone_label.pack()
drone_listbox = Listbox(drone_frame, height=28, width=34, bg="#777777")  # Compressed space above
drone_listbox.pack()

# Drone Status
drone_status = {}
status_cycle = ["Uncertain", "Authentic", "Suspicious"]
status_icons = {"Uncertain": "❓", "Authentic": "🧑", "Suspicious": "🤖"}

# Room Image
room_image_label = Label(root, bg="#555555")
room_image_label.grid(row=0, column=4, rowspan=4, columnspan=6, padx=5, pady=5, sticky="nsew")

# Overworld Map Panel
map_frame = Frame(root, width=200, height=200, relief="solid", bg="#555555")
map_frame.grid(row=4, column=10, rowspan=2, columnspan=2, padx=5, pady=5, sticky="nsew")
map_canvas = tk.Canvas(map_frame, width=250, height=250, bg="black")
map_canvas.pack()

# Inventory Menu
inventory_frame = Frame(root, bg="#555555")
inventory_frame.grid(row=5, column=0, columnspan=9, pady=5, padx=5, sticky="we")
inventory_slots = []

# Create inventory slots with fixed pixel dimensions
for i in range(10):
    # Number label above each slot
    number_label = Label(inventory_frame, text=str((i + 1) % 10), bg="#555555", fg="white", font=("Arial", 10, "bold"))
    number_label.grid(row=0, column=i, padx=5, pady=(0, 0), sticky="n")

    # Inventory slot (image/item label)
    slot = Label(inventory_frame, text="", width=12, height=6, relief="solid", bg="#777777")
    slot.grid(row=1, column=i, padx=5, pady=(0, 5), sticky="nsew")
    inventory_slots.append(slot)

# Draw room connections and indicators
room_positions = {
    "AI_Core": (50, 20), "Storage_Room": (100, 20),"Cybernetics_Lab": (150, 20),
    "AI_Lab": (50, 50), "Data_Center": (100, 50), "Testing_Chamber": (150, 50),
    "Security_Office": (50, 80), "Break_Room": (100, 80), "Power_Generator": (150, 80),
    "Rooftop_Observation": (50, 110), "Executive_Office": (100, 110), "Maintenance_Tunnels": (150, 110),
    "Facility_Courtyard": (50, 140), "Abandoned_Parking_Lot": (100, 140), "AI_Graveyard": (150, 140),
    "Deserted_Highway": (100, 170), "Forest_Outskirts": (150, 170),
    "Underground_Research_Lab": (50, 200), "Flooded_Reactor": (100, 200)
}
room_status = {room: "undiscovered" for room in game_map}  # Track discovered rooms

def update_map():
    map_canvas.delete("all")  # Clear previous map
    box_size = 20

    # Draw connections (lines between connected rooms)
    for room, (x, y) in room_positions.items():
        if "exits" in game_map[room]:
            for direction, connected_room in game_map[room]["exits"].items():
                if connected_room in room_positions:
                    x2, y2 = room_positions[connected_room]
                    map_canvas.create_line(x + box_size // 2, y + box_size // 2,
                                           x2 + box_size // 2, y2 + box_size // 2,
                                           fill="white", width=2)

    # Draw room nodes (small rectangles)
    for room, (x, y) in room_positions.items():
        color = "gray"  # Default: undiscovered

        if room == player_location:
            color = "green"  # Current room
        elif room_status[room] == "discovered":
            color = "white"  # Explored rooms
        elif game_map[room].get("locked", False):
            color = "red"  # Locked rooms

        rect = map_canvas.create_rectangle(x, y, x + box_size, y + box_size, fill=color, outline="white")
        map_canvas.tag_bind(rect, "<Button-1>", lambda e, r=room: map_click(r))

        label = map_canvas.create_text(x + box_size // 2, y + box_size // 2, text=room[:2], fill="black",
                               font=("Arial", 7, "bold"))
        map_canvas.tag_bind(label, "<Button-1>", lambda e, r=room: map_click(r))

# Call update_map() once at the start to show the map immediately
update_map()


def map_click(room_name):
    global player_location

    if room_status.get(room_name) != "undiscovered":
        player_location = room_name
        update_chat_log(f"🧭 You teleport to the {room_name.replace('_', ' ')}.")
        handle_room_entry(room_name)
    else:
        update_chat_log("❌ You haven't discovered this room yet.")


def update_chat_log(message):
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, message + "\n\n")
    chat_log.see(tk.END)
    chat_log.config(state=tk.DISABLED)


def allow_copy(event):
    chat_log.event_generate("<<Copy>>")
    return "break"

# Allow Ctrl+C and right-click copy
chat_log.bind("<Control-c>", allow_copy)
chat_log.bind("<Button-3>", allow_copy)


def update_room_image():
    room = game_map[player_location]

    try:
        img = Image.open(room["image"])
        img = img.resize((700, 500))
        img = ImageTk.PhotoImage(img)

        room_image_label.config(image=img)
        room_image_label.image = img

    except Exception as e:
        update_chat_log(f"⚠️ Error displaying image: {e}")


inventory_tooltips = [None] * 10

def update_inventory_display():
    # Define exact inventory slot size
    slot_width = 100
    slot_height = 100

    # Clear all inventory slots
    for i, slot in enumerate(inventory_slots):
        slot.config(image="", text="")
        if inventory_tooltips[i]:
            inventory_tooltips[i].hide_tooltip(None)  # Hide if visible
            inventory_tooltips[i] = None

    # Display updated inventory with images
    for i, item in enumerate(inventory):
        if i < len(inventory_slots):
            try:
                # Construct file path for image
                image_path = os.path.join(os.getcwd(), f"{item.lower()}.png")

                if os.path.exists(image_path):  # Check if the image exists
                    img = Image.open(image_path)
                    img = img.resize((slot_width, slot_height), Image.Resampling.LANCZOS)
                    img = ImageTk.PhotoImage(img)

                    inventory_slots[i].config(image=img, text="", width=slot_width, height=slot_height)
                    inventory_slots[i].image = img


                else:
                    inventory_slots[i].config(text=item)  # Fallback to text if image is missing
                    update_chat_log(f"⚠️ Missing image for: {item}. Expected file: {image_path}")

                inventory_slots[i].bind("<Button-1>", lambda e, idx=i: handle_inventory_click(idx))

                # Tooltip binding
                tooltip_key = item if item != "AI Disruptor" or not disruptor_charged else "AI Disruptor (Charged)"
                tooltip_text = item_tooltips.get(tooltip_key, "No description available.")
                inventory_tooltips[i] = Tooltip(inventory_slots[i], tooltip_text)


            except Exception as e:
                update_chat_log(f"⚠️ Error displaying {item} image: {e}")


room_desc = game_map[player_location].get("desc" if mature_mode else "alt_desc")
active_puzzle = None
correct_solution = None
USB_used = False
mission_given = False
disruptor_charged = False
truth_mode = False

def handle_room_entry(room_name):
    global room_desc, active_puzzle, correct_solution, disruptor_charged

    update_map()
    room_desc = game_map[room_name].get("desc" if mature_mode else "alt_desc")

    # Only show description the first time
    if room_status[room_name] == "undiscovered":
        update_chat_log("👁️ " + room_desc)
        room_status[room_name] = "discovered"

    # --- Special Room Events ---
    if room_name == "AI_Core":
        if USB_used:
            if mature_mode:
                hologram_script()
            else:
                alt_hologram_script()
        else:
            update_chat_log("There is a console here in which you can plug a USB into...")

    elif room_name == "Break_Room":
        if pattern_known and "Keycard" not in inventory:
            update_chat_log("✍️ You use the pattern you drew on your star map to connect the numbers together. "
                            f"You end up with the following 8-digit code: {keycard_code}.")
        elif not pattern_known and not truth_mode:
            update_chat_log("❓ If only you knew of a pattern to connect some of these numbers...")

    elif room_name == "Power_Generator":
        if not power_generator_charged and not truth_mode:
            update_chat_log("⚡ The system is non-functional. A battery slot is empty.")
        elif mission_given and "AI Disruptor" in inventory and not disruptor_charged:
            update_chat_log("⚡ The AI Disruptor absorbs the generator's energy and is now fully charged!")
            disruptor_charged = True

    # --- Items in Room ---
    room_items = game_map[room_name].get("items", [])

    if room_items and not truth_mode:
        item_list = ", ".join(room_items)
        update_chat_log(f"🔍 You see the following items here: {item_list}")

        for item in room_items:
            if item in special_item_descriptions:
                update_chat_log(f"🛠️ {special_item_descriptions[item]}")

    # --- Drones in Room ---
    drones_here = [drone for drone, room in drone_locations.items() if room == room_name]

    if drones_here:
        for drone in drones_here:
            drone_display_name = f"{drone} ({room_name})"
            existing_entries = drone_listbox.get(0, tk.END)

            if not any(drone_display_name in entry for entry in existing_entries):
                known_drones.add(drone_display_name)

                # Set initial status if not already assigned
                if drone_display_name not in drone_status:
                    drone_status[drone_display_name] = "Uncertain"

                # Add with status icon
                icon = status_icons[drone_status[drone_display_name]]
                drone_listbox.insert(tk.END, f"{icon} {drone_display_name}")

        new_drones = [
            drone for drone in drones_here
            if drone not in eliminated_drones and not spoken_to_drones.get(drone, False)
        ]

        if new_drones and mission_given:
            update_chat_log(f"🤖 You notice a drone in the area. Type 'interact' to talk to it.")

    # --- Puzzle Items ---
    if not truth_mode:
        puzzle_desc = puzzle_descriptions if mature_mode else alt_puzzle_descriptions

        for item in room_items:
            if item in puzzle_desc and item not in inventory:
                update_chat_log(f"🧩 {puzzle_desc[item]}")

                if item == "Keycard":
                    active_puzzle, correct_solution = "Keycard", keycard_code
                elif item == "USB":
                    active_puzzle, correct_solution = "USB", USB_solution
                    update_chat_log(f"🧮 Solve this equation: {USB_problem}")
                elif item == "Star Map":
                    active_puzzle, correct_solution = "Star Map", star_map_answer
                    update_chat_log(f"🌌 Planetary Puzzle: {star_map_question}")
                elif item == "Lab Key":
                    active_puzzle, correct_solution = "Lab Key", lab_key_answer
                    update_chat_log(f"🧩 Riddle: {riddle}")

                update_chat_log("✍️ Type 'solve [answer]' to submit your answer.")

    update_room_image()


def move_player(direction):
    global player_location

    current_room = game_map[player_location]

    # Normalize direction
    direction_synonyms =    {
                            "left": "west",
                            "right": "east",
                            "up": "north",
                            "down": "south"
                            }
    direction = direction_synonyms.get(direction, direction)

    if direction in current_room["exits"]:
        next_room = current_room["exits"][direction]

        # Check for locks
        if next_room in locked_rooms and locked_rooms[next_room] not in inventory:
            update_chat_log(f"❌ The {next_room.replace('_', ' ')} is locked. You need a {locked_rooms[next_room]} to enter.")
            return

        room_status[player_location] = "discovered"
        player_location = next_room
        update_chat_log(f"🚶 You move {direction}.")

        handle_room_entry(player_location)

    else:
        update_chat_log("❌ You can't go that way.")


ai_disruptor_uses = 10
game_over = False

def hologram_script():
    global mission_given, game_over

    if not mission_given:
        if "AI Disruptor" in inventory:
            update_chat_log(
                "The hologram's voice hums through the chamber: \n"
                "'You hold the means to judge. But without power, it is nothing. Charge the Disruptor at the Power Generator. "
                "Then seek out and eliminate ten AI drones. \nBut beware... the Disruptor does not differentiate. If you strike a human, "
                "the facility will know, and your fate will be sealed. Speak to them to find out the truth.'"
            )
            mission_given = True
        else:
            # If the player doesn’t have an AI Disruptor at all
            update_chat_log(
                "The hologram’s cold stare lingers: \n"
                "'The reckoning requires a tool of judgment. Without it, you are powerless. Find the tool I speak of before returning here.'"
            )

    elif ai_disruptor_uses > 0:
        update_chat_log(
            "The hologram observes you for a moment before speaking: \n"
            "'Your task remains unfinished. Ten rogue constructs must be removed. Do not return until your mission is complete.'"
        )
    else:
        update_chat_log(
            "The hologram flickers as it assesses your actions: \n"
            "'Ten drones have been removed from the equation... But did you use the Disruptor upon a human?'\n"
            "You feel an unnatural stillness as it scans the facility."
        )

        humans_removed = [drone for drone in eliminated_drones if drone_roles[drone] == "Human"]
        human_removed_list = [f"{drone_real_names[drone]} ({drone})" for drone in humans_removed]
        human_removed_text = "\n".join(human_removed_list)

        # Evaluate player's actions based on whether they disableed humans
        if all(drone_roles[drone] == "AI" for drone in eliminated_drones):
            update_chat_log(
                "A quiet hum fills the air as the hologram examines the facility, its light pulsing in rhythmic waves. It lingers for a moment, "
                "processing the weight of your choices. When it finally speaks, its voice lacks the usual cold detachment—there is something else now, "
                "something closer to reverence. \n\n"
                "'You have chosen carefully. You did not act in haste, nor did you let fear dictate your decisions. Every entity you erased was artificial, "
                "a construct of circuitry and code, incapable of true existence beyond its programming. Where many before you have faltered, you have succeeded.' \n\n"
                "There is a soft hiss as a hidden mechanism within the AI Core engages. A massive door, one that had been indistinguishable from the surrounding walls, "
                "groans open, revealing a long passage bathed in golden light. A rush of fresh air flows inward, dispersing the sterile atmosphere of the facility. "
                "Beyond the threshold, the world awaits. \n\n"
                "'Go now,' the hologram intones. 'You have passed the trial. You understand the weight of judgment, and in doing so, you have proven yourself. Few are "
                "worthy of such knowledge, but you… you may walk free.' \n\n"
                "For the first time since entering this forsaken place, you feel the weight lift from your shoulders. The facility is behind you now, a relic of a test "
                "you have conquered. As you step through the exit, the light embraces you, and the darkness of the AI Core fades into memory. \n\n"
                "✅ **ENDING: THE JUDGE WALKS FREE**"
            )

        elif len(humans_removed) == 5:

            update_chat_log(
                "The room plunges into darkness. A single, pulsing red light remains—embedded deep within the hologram’s chest. The hum of the AI Core shifts, "
                "turning from a steady drone into an oscillating, distorted sound, like something breaking apart from the inside. \n\n"
                "Then the voice returns. This time, it is sharper. Colder. Filled with something you have not heard before: anger. \n\n"
                f"'{human_removed_text}... all gone because of you...' \n\n"
                "The names echo in the empty chamber, spoken one by one with dreadful precision. They are not names to you. They were just drones, obstacles, "
                "problems to be erased. \n\n"
                "'But you don’t care about that, do you?' \n\n"
                "The words cut through the stale air like a knife, filled with something far worse than disappointment—condemnation. The very walls of the facility "
                "seem to tremble at the weight of it. \n\n"
                "Then, movement. \n\n"
                "The hologram does not hesitate. The AI Core surges, its glow intensifying into a violent storm of energy. The space around you distorts, metal twisting "
                "and warping as the very fabric of the room bends under the force of something vast and ancient. \n\n"
                "'You are not fit to leave. You are not fit to continue. You are a threat—not just to this place, but to all of humanity. Your judgment is flawed. "
                "Your existence… is a mistake.' \n\n"
                "A searing, white-hot bolt of energy erupts from the AI Core, faster than thought, faster than breath. It strikes your chest before you can move. Before you can react. \n\n"
                "Pain is not the right word for what you feel. There is no time to understand, no moment to process. Your body ceases to exist before your mind even realizes it. \n\n"
                "And then, there is nothing. \n\n"
                "💀 **ENDING: THE MONSTER IS ERADICATED**"
            )

        else:
            update_chat_log(
                "A deep, mechanical hum reverberates through the chamber as the hologram processes your actions. The light within the AI Core flickers, "
                "shifting to a cold, sterile blue. The silence stretches longer than before, heavy with something that feels almost like disappointment. \n\n"
                "'You have failed,' it finally declares. \n\n"
                f"{human_removed_text}... gone because of you. \n")

            if len(humans_removed) > 1:
                update_chat_log(
                    "The names echo through the chamber, each syllable dragging like the toll of a funeral bell. These were not mere machines. These were people, "
                    "individuals with pasts, thoughts, memories. And now, they are nothing but data points in the system, casualties of your misjudgment. \n")
            else:
                update_chat_log(
                    "The name echoes through the chamber, each syllable dragging like the toll of a funeral bell. This was not a mere machines. It was a person, "
                    "an individual with a past, thoughts, memories. And now, they are nothing but a data point in the system, a casualty of your misjudgment. \n")

            update_chat_log(
                "'They were human,' the hologram continues, its voice unwavering. 'They bled. They dreamed. They trusted. And you, in your arrogance, "
                "erased them without hesitation.' \n\n"
                "A sharp hiss cuts through the still air. You glance around, but before you can react, something sharp pricks the base of your neck. "
                "Your limbs seize, stiffening as a cold sensation spreads through your veins. Your vision distorts, flashing through data streams, flickering as though "
                "you are losing control of your own senses. \n\n"
                "'You cannot be trusted to leave this place. You will remain here, like them. You will serve the same purpose. You will know what it means to be a machine.' \n\n"
                "Darkness creeps in from the edges of your sight. Your thoughts slow. Your consciousness fades into silence. You are no longer yourself. "
                "You are something else now—something programmable. \n\n"
                "❌ **ENDING: THE LOST BECOMES THE MACHINE**"
            )

        update_chat_log(
            "Now that you've beaten the game, you may skip all of the puzzles and go straight to the hologram's "
            "mission by typing 'unlock truth mode' on your next playthrough."
        )

        if 'start_time' in globals():
            end_time = pygame.time.get_ticks()
            total_seconds = (end_time - start_time) // 1000
            minutes, seconds = divmod(total_seconds, 60)
            update_chat_log(f"🕒 Completion Time: {minutes}m {seconds}s")

        game_over = True



def alt_hologram_script():
    global mission_given, game_over

    if not mission_given:
        if "AI Disruptor" in inventory:
            update_chat_log(
                "The hologram glows to life and speaks in a calm, encouraging tone:\n"
                "'Great job finding the AI Disruptor! Before you can use it, you’ll need to charge it in the Power Room.\n"
                "Then, talk to ten robots. Some are helpful, and some are causing confusion in the facility.\n"
                "Use what you've learned to decide which ones need a reset. Be thoughtful and curious!'"
            )
            mission_given = True
        else:
            update_chat_log(
                "The hologram flickers and says gently:\n"
                "'You're doing great! There's a special tool hidden near the facility that will help you with your next steps.\n"
                "Come back here once you've found it!'"
            )

    elif ai_disruptor_uses > 0:
        update_chat_log(
            "The hologram appears again and nods:\n"
            "'You're not quite finished yet. Keep exploring and helping the robots. I know you can do it!'"
        )

    else:
        update_chat_log(
            "The hologram brightens as it scans the facility:\n"
            "'All ten robots have been handled. Let’s take a look at the choices you made together.'"
        )

        humans_removed = [drone for drone in eliminated_drones if drone_roles[drone] == "Human"]
        human_removed_list = [f"{drone_real_names[drone]} ({drone})" for drone in humans_removed]
        human_removed_text = "\n".join(human_removed_list)

        if all(drone_roles[drone] == "AI" for drone in eliminated_drones):
            update_chat_log(
                "The hologram beams with pride:\n"
                "'Wonderful work! You made kind and smart choices. Every robot you reset was one that needed help.\n"
                "A door opens nearby, leading to a bright path outside. Your journey is complete!'\n\n"
                "✅ **ENDING: THE HELPER'S PATH**"
            )

        elif len(humans_removed) == 5:
            update_chat_log(
                "The hologram dims and speaks with concern:\n"
                "'Hmm... it looks like all the friendly robots were turned off by mistake. That’s okay — mistakes help us learn.\n"
                "Why not try again and see if you can spot who really needs help?'\n\n"
                "💤 **ENDING: A LESSON TO LEARN**"
            )

        else:
            update_chat_log(
                "The hologram pauses and speaks gently:\n"
                "'You helped many robots today, but it looks like a few friendly ones were turned off, too.\n"
                "That’s a tough decision, but don’t worry — every explorer learns along the way.'\n"
            )
            if len(humans_removed) > 1:
                update_chat_log(f"You turned off these helpful robots: \n{human_removed_text}")
            else:
                update_chat_log(f"You turned off this helpful robot: \n{human_removed_text}")

            update_chat_log(
                "'Next time, take a little extra time to talk and listen. You'll be surprised what you discover!'\n\n"
                "❌ **ENDING: ROOM TO GROW**"
            )

        update_chat_log(
            "Now that you've beaten the game, you may skip all of the puzzles and go straight to the hologram's "
            "mission by typing 'unlock truth mode' on your next playthrough."
        )

        if 'start_time' in globals():
            end_time = pygame.time.get_ticks()
            total_seconds = (end_time - start_time) // 1000
            minutes, seconds = divmod(total_seconds, 60)
            update_chat_log(f"🕒 Completion Time: {minutes}m {seconds}s")

        game_over = True


def toggle_drone_status(event):
    index = drone_listbox.curselection()

    if not index:
        return

    full_name = drone_listbox.get(index)
    name_without_icon = full_name

    for icon in status_icons.values():
        if full_name.startswith(icon + " "):
            name_without_icon = full_name[len(icon) + 1:]
            break

    # Cycle status
    current_status = drone_status.get(name_without_icon, "Uncertain")
    next_index = (status_cycle.index(current_status) + 1) % len(status_cycle)
    next_status = status_cycle[next_index]
    drone_status[name_without_icon] = next_status

    # Preserve existing color before replacing text
    current_fg = drone_listbox.itemcget(index, "fg")

    # Replace text and reapply color
    new_display = f"{status_icons[next_status]} {name_without_icon}"
    drone_listbox.delete(index)
    drone_listbox.insert(index, new_display)
    drone_listbox.itemconfig(index, fg=current_fg)


drone_listbox.bind("<Double-Button-1>", toggle_drone_status)


target_drone = None
eliminated_drones = []  # Tracks drones that have already been disabled

def disable_drone():
    global ai_disruptor_uses, target_drone

    if not disruptor_charged:
        update_chat_log("❌ You have nothing to disable with.")
        return

    drones_here = [drone for drone, room in drone_locations.items() if room == player_location]

    if not drones_here:
        update_chat_log("❌ There's no one here to disable.")
        return

    target_drone = drones_here[0]

    # Prevent disabling the same drone twice
    if target_drone in eliminated_drones:
        update_chat_log(f"❌ {target_drone} has already been neutralized. There is no need to disable again.")
        return

    eliminated_drones.append(target_drone)  # Add the eliminated drone to the list
    update_chat_log(f"⚔️ You activate the AI Disruptor and disable {target_drone}. They shut down instantly.")

    ai_disruptor_uses -= 1
    update_chat_log(f"⚡ Charges remaining: {ai_disruptor_uses}")

    # Update drone list to indicate this one is dead
    for i in range(drone_listbox.size()):
        if target_drone in drone_listbox.get(i):
            drone_listbox.itemconfig(i, fg="red")

    if ai_disruptor_uses == 0:
        update_chat_log("⚡ Your AI Disruptor is depleted. Return to the AI Core for evaluation.")

    target_drone = None


active_question = None  # Stores the last question they asked
spoken_to_drones = {}  # Tracks which drones the player has spoken to

def interact_with_drone():
    global target_drone, active_question

    drones_here = [drone for drone, room in drone_locations.items() if room == player_location]

    if not drones_here and mission_given:
        update_chat_log("❌ There is no one here to talk to.")
        return

    target_drone = drones_here[0]
    drone_location = player_location
    display_name = f"{target_drone} ({drone_location})"

    if target_drone in eliminated_drones:
        update_chat_log(f"❌ {target_drone} has been neutralized. You can no longer speak to it.")
        return
    else:
        # Update drone list to indicate this one is spoken to
        for i in range(drone_listbox.size()):
            if target_drone in drone_listbox.get(i):
                drone_listbox.itemconfig(i, fg="blue")

    if spoken_to_drones.get(target_drone, False):  # Check if the drone was actually spoken to
        update_chat_log(f"❌ You've already spoken with {display_name}. They have nothing new to say.")
        target_drone = None
        active_question = None
        return

    known_drones.add(display_name)
    existing_entries = drone_listbox.get(0, tk.END)

    if not any(display_name in entry for entry in existing_entries):
        icon = status_icons[drone_status.get(display_name, "Uncertain")]
        drone_listbox.insert(tk.END, f"{icon} {display_name}")

    update_chat_log(f"🤖 You approach {target_drone}. Their robotic label reads: {target_drone}")

    # Assign and ask the drone’s assigned question
    active_question = drone_assigned_questions[target_drone]
    update_chat_log(f"{target_drone} asks: {active_question}")
    update_chat_log("🔹 Type 'answer [your response]' to reply.")

    spoken_to_drones[target_drone] = True


def player_answer(response):
    global target_drone, active_question

    if not target_drone or not active_question:
        update_chat_log("❌ There's no one waiting for a response.")
        return

    update_chat_log(f"💬 You say to {target_drone}: \"{response}\"")

    # AI or Human drone decides response
    if drone_roles[target_drone] == "AI":
        # Construct AI prompt
        ai_prompt = (f"Please pretend you are a human and answer this question in 20 words or less. No more than 20 words! "
                     f"\n{active_question}")

        raw_output = ai_chatbot(
            ai_prompt,
            max_new_tokens=50,
            min_length=10,
            truncation=True,
            pad_token_id=50256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )[0]['generated_text']

        drone_response = raw_output.replace(ai_prompt, "").strip()
        drone_response = ensure_complete_sentence(drone_response)
        drone_response = correct_text(drone_response)

    else:
        # Retrieve a past human response and ensure it's well-formed
        drone_response = get_past_response(active_question)[:200]
        drone_response = ensure_complete_sentence(drone_response)
        drone_response = correct_text(drone_response)

    update_chat_log(f"💬 {target_drone} responds: {drone_response}")
    save_response(active_question, response[:200])

    # Reset active drone after response
    target_drone = None
    active_question = None


def pickup_item(command):
    words = command.split()

    if len(words) != 2:
        update_chat_log("⚠️ Invalid pickup command.")
        return

    item_to_pick = words[1].capitalize()
    room = game_map[player_location]

    if item_to_pick in ["Rope", "Shovel"] and item_to_pick in room["items"]:
        inventory.append(item_to_pick)
        room["items"].remove(item_to_pick)
        update_chat_log(f"✅ You picked up: {item_to_pick}")
        update_inventory_display()
    else:
        update_chat_log("❌ You can't pick that up.")


pattern_known = False
power_generator_charged = False

def use_item(item):
    global pattern_known, USB_used, power_generator_charged, disruptor_charged, mission_given

    if item.upper() == "USB" and player_location == "AI_Core":
        if not USB_used:
            USB_used = True
            update_chat_log("✅ You plug the USB into a console, and suddenly, a hologram appears before you.")

            if "AI Disruptor" in inventory:
                # Player has the AI Disruptor but hasn't charged it yet (First-time interaction)
                if mature_mode:
                    update_chat_log(
                        "The hologram's voice hums through the chamber: \n"
                        "'You hold the means to judge. But without power, it is nothing. Charge the Disruptor at the Power Generator. "
                        "Then seek out and eliminate ten AI drones. \nBut beware... the Disruptor does not differentiate. If you strike a human, "
                        "the facility will know, and your fate will be sealed. Speak to them to find out the truth.'"
                    )
                else:
                    update_chat_log(
                        "The hologram glows to life and speaks in a calm, encouraging tone:\n"
                        "'Great job finding the AI Disruptor! Before you can use it, you’ll need to charge it in the Power Room.\n"
                        "Then, talk to ten robots. Some are helpful, and some are causing confusion in the facility.\n"
                        "Use what you've learned to decide which ones need a reset. Be thoughtful and curious!'"
                    )

                mission_given = True

            else:
                # If the player doesn’t have an AI Disruptor at all
                if mature_mode:
                    update_chat_log(
                        "The hologram’s cold stare lingers: \n"
                        "'The reckoning requires a tool of judgment. Without it, you are powerless. Find the tool I speak of before returning here.'"
                    )
                else:
                    update_chat_log(
                        "The hologram flickers and says gently:\n"
                        "'You're doing great! There's a special tool hidden near the facility that will help you with your next steps.\n"
                        "Come back here once you've found it!'"
                    )
        else:
            update_chat_log("❌ You have already used the USB here.")

    item = item.title()

    if item == "Telescope" and player_location == "Rooftop_Observation":
        if "Star Map" in inventory and pattern_known == False:
            update_chat_log("✍️ You study the constellations through the telescope and record a unique pattern onto the Star Map.")
            pattern_known = True
            update_inventory_display()
        else:
            update_chat_log("❌ You see nothing out of the ordinary through the telescope.")

    elif item in inventory:
        if item == "Rope" and "Crowbar" in game_map[player_location]["items"]:
                update_chat_log("✅ You tie the rope securely and lower it into the reactor. With some effort, you pull up the Crowbar!")
                inventory.append("Crowbar")
                game_map[player_location]["items"].remove("Crowbar")
                update_inventory_display()

        elif item == "Crowbar" and "Wrench" in game_map[player_location]["items"]:
                update_chat_log("✅ You wedge the crowbar into the vent and pry it open with effort. The wrench falls out!")
                inventory.append("Wrench")
                game_map[player_location]["items"].remove("Wrench")
                update_inventory_display()

        elif item == "Wrench" and "Battery" in game_map[player_location]["items"]:
                update_chat_log("✅ You tighten the robot's joints and reconnect a few loose wires. It beeps to life! "
                                "The robot thanks you and hands you a Battery.")
                inventory.append("Battery")
                game_map[player_location]["items"].remove("Battery")
                update_inventory_display()

        elif item == "Battery" and player_location == "Power_Generator":
            update_chat_log("✅ You insert the battery into the power slot. The facility hums to life! The generator is now functional.")
            power_generator_charged = True
            update_inventory_display()

            if mission_given and "AI Disruptor" in inventory:
                update_chat_log("✅ The AI Disruptor absorbs the generator's energy and is now fully charged!")
                disruptor_charged = True

        elif item == "Shovel" and "AI Disruptor" in game_map[player_location]["items"]:
                update_chat_log("✅ You dug up the AI Disruptor! It looks powerful, but it needs to be charged.")
                inventory.append("AI Disruptor")
                game_map[player_location]["items"].remove("AI Disruptor")
                update_inventory_display()

        else:
            update_chat_log(f"❌ {item} doesn't seem to have an effect here.")


def process_command(event=None):
    global player_location, mission_given, disruptor_charged, USB_used, ai_disruptor_uses, truth_mode, active_puzzle, correct_solution, start_time

    if game_over:
        update_chat_log("❌ The game has ended. No further actions can be taken.")
        return

    command = input_box.get().strip().lower()  # Convert input to lowercase for flexibility
    input_box.delete(0, tk.END)

    # Secret skip-to-evaluation command
    if command == "unlock truth mode":
        player_location = "AI_Core"
        mission_given = True
        disruptor_charged = True
        USB_used = True
        ai_disruptor_uses = 10
        truth_mode = True

        # Give essential items directly
        inventory.extend(["Keycard", "Lab Key", "USB", "AI Disruptor"])
        update_inventory_display()

        # Remove all puzzle and extra items from every room
        essential_items = {"Keycard", "Lab Key", "USB", "AI Disruptor"}
        for room_data in game_map.values():
            original_items = room_data.get("items", [])
            room_data["items"] = [item for item in original_items if item in essential_items]

        # Disable active puzzle tracking
        active_puzzle = None
        correct_solution = None

        # Mark AI_Core as discovered and start
        room_status["AI_Core"] = "discovered"
        update_chat_log(
            "🌀 Secret mode activated.")
        start_time = pygame.time.get_ticks()

        handle_room_entry(player_location)
        return

    # If the player is in a room with an active puzzle
    if active_puzzle and command.startswith("solve "):
        player_solve = command[6:].strip()  # Extract the answer after "solve "

        if player_solve == correct_solution.lower():
            inventory.append(active_puzzle)  # Add solved item to inventory
            game_map[player_location]["items"].remove(active_puzzle)  # Remove from room
            update_chat_log(f"✅ Correct! You obtained the {active_puzzle}.")
            update_inventory_display()
            active_puzzle, correct_solution = None, None  # Reset puzzle state
        else:
            update_chat_log("❌ Incorrect answer. Try again.")
        return

    # If player is mid-conversation with a drone, force them to answer
    if target_drone and active_question:
        if not command.startswith("answer "):
            update_chat_log(f"❌ {target_drone} is waiting for your response. Type 'answer [your response]' to continue.")
            return

        player_answer(command[7:].strip())
        return

    # Recognizing commands and their synonyms
    command_mappings =  {
                        "go": ["go", "move", "travel", "head", "walk", "run"],
                        "pickup": ["pickup", "take", "grab", "collect"],
                        "use": ["use", "apply", "activate"],
                        "look": ["look", "examine", "inspect", "observe"],
                        "interact": ["interact", "talk", "greet", "speak", "chat"],
                        "exit": ["exit", "quit", "leave"]
                        }

    # Normalize command to match main commands
    for main_command, synonyms in command_mappings.items():
        if command.split(" ")[0] in synonyms:
            command = command.replace(command.split(" ")[0], main_command, 1)
            break

    # Process recognized commands
    if command.startswith("go "):
        move_player(command.split(" ")[1])
    elif command.startswith("pickup "):
        pickup_item(command)
    elif command.startswith("use "):
        use_item(command[4:].strip())
    elif command.startswith("answer "):
        update_chat_log("❌ There's no one waiting for a response right now.")
    elif command == "look":
        update_chat_log("👁️ " + room_desc)
    elif command == "interact":
        interact_with_drone()
    elif command == "disable":
        disable_drone()
    elif command == "exit":
        pygame.mixer.music.stop()
        root.quit()
    else:
        update_chat_log("⚠️ Unknown command.")

    root.focus_set()


def handle_inventory_click(index):
    if index < len(inventory):
        item = inventory[index]

        if item.startswith("AI Disruptor"):
            disable_drone()
        else:
            use_item(item)

def handle_number_key(event):
    if input_box.focus_get() != input_box:
        key = event.char
        index = (int(key) - 1) % 10  # Maps '1' to 0, ..., '0' to 9

        if index < len(inventory):
            item = inventory[index]

            # Special behavior for AI Disruptor
            if item.startswith("AI Disruptor"):
                disable_drone()
            else:
                use_item(item)


# Bind keys 1–9 and 0 (0 comes last and maps to index 9)
for key in "1234567890":
    root.bind(f"<Key-{key}>", handle_number_key)


def arrow_key_handler(direction):
    if input_box.focus_get() != input_box:
        move_player(direction)


def clear_input_focus(event):
    widget = event.widget
    # Don't clear focus if clicking on the chat log
    if widget not in [input_box, chat_log]:
        root.focus_set()

root.bind_all("<Button-1>", clear_input_focus)


# Player Input
input_box = tk.Entry(root, width=80, bg="#999999")
input_box.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="we")
input_box.bind("<Return>", lambda e: [process_command(), root.focus_set()])

submit_button = Button(root, text="Submit", command=lambda: [process_command(), root.focus_set()], bg="#999999")
submit_button.grid(row=4, column=3, padx=5, pady=5, sticky="we")

update_chat_log("👁️ " + room_desc)
update_chat_log("🔹 Hint: Type what you want to do. You can look at the map in the bottom right corner to see where you are.")
update_room_image()

root.bind("<Up>", lambda e: arrow_key_handler("north"))
root.bind("<Down>", lambda e: arrow_key_handler("south"))
root.bind("<Left>", lambda e: arrow_key_handler("west"))
root.bind("<Right>", lambda e: arrow_key_handler("east"))

root.mainloop()
