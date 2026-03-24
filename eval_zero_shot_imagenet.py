"""
Zero-shot ImageNet-1k evaluation for ShareLock models.

Follows CLIP-Benchmark zero-shot evaluation approach:
  1. Build class prototypes by encoding ImageNet class names with 80 CLIP templates
  2. Encode validation images with DINOv2 + vision projector
  3. Classify via cosine similarity, report top-1 and top-5 accuracy

Prerequisites:
    # Download ImageNet-1k validation set (Parquet, ~6.5 GB):
    huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset \
        --include 'data/validation*' --local-dir datasets/imagenet-1k

    # Download checkpoint (CC3M, expected ~54.5% top-1):
    python -c "
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id='FunAILab/ShareLock', filename='ShareLock-CC3M.ckpt', local_dir='checkpoints')
    "

Usage:
    python eval_zero_shot_imagenet.py --checkpoint checkpoints/ShareLock-CC3M.ckpt

    # With single custom template:
    python eval_zero_shot_imagenet.py --checkpoint checkpoints/ShareLock-CC3M.ckpt --template "a photo of a {}"
"""

import torch
import torch.serialization
import argparse
import glob as glob_module
import os
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig, ListConfig
from datasets import load_dataset
from datasets import Image as HFImage
from torch.utils.data import DataLoader

# PyTorch 2.6+ defaults weights_only=True, but PL checkpoints contain OmegaConf objects.
# Patch torch.load so that PyTorch Lightning's internal loader uses weights_only=False.
_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # force: PL checkpoints contain OmegaConf objects
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from torchvision import transforms as tv_transforms
from sharelock.models.model import ShareLock


# 80-template ensemble from the CLIP paper / CLIP-Benchmark
# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


# OpenAI / DINOv2 curated ImageNet class names (single clean name per class).
# These are used in the DINOv2 dinotxt notebook and match CLIP-Benchmark evaluation.
# Source: https://github.com/facebookresearch/dinov2/blob/main/notebooks/dinotxt.ipynb
IMAGENET_OPENAI_CLASSNAMES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch",
    "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay",
    "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle",
    "vulture", "great grey owl", "fire salamander", "smooth newt", "newt",
    "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog",
    "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard",
    "alligator lizard", "Gila monster", "European green lizard", "chameleon",
    "Komodo dragon", "Nile crocodile", "American alligator", "triceratops",
    "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake",
    "kingsnake", "garter snake", "water snake", "vine snake", "night snake",
    "boa constrictor", "African rock python", "Indian cobra", "green mamba",
    "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion",
    "yellow garden spider", "barn spider", "European garden spider",
    "southern black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar",
    "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker",
    "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone",
    "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug",
    "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab",
    "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo",
    "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin",
    "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin",
    "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin",
    "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua",
    "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound",
    "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound",
    "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi",
    "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound",
    "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner",
    "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier",
    "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer",
    "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier",
    "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso",
    "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer",
    "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog",
    "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz",
    "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie",
    "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler",
    "German Shepherd Dog", "Dobermann", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund",
    "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute",
    "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger",
    "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow",
    "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi",
    "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog",
    "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat",
    "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx",
    "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear",
    "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant",
    "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis",
    "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly",
    "red admiral butterfly", "ringlet butterfly", "monarch butterfly",
    "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare",
    "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver",
    "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog",
    "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)",
    "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle",
    "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo",
    "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
    "guenon", "patas monkey", "baboon", "macaque", "langur",
    "black-and-white colobus", "proboscis monkey", "marmoset",
    "white-headed capuchin", "howler monkey", "titi monkey",
    "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur",
    "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda",
    "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier",
    "airliner", "airship", "altar", "ambulance", "amphibious vehicle",
    "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack",
    "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn",
    "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet",
    "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon",
    "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle",
    "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie",
    "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie",
    "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket",
    "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab",
    "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan",
    "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel",
    "automated teller machine", "cassette", "cassette player", "castle",
    "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence",
    "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime",
    "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver",
    "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug",
    "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet",
    "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet",
    "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass",
    "dam", "desk", "desktop computer", "rotary dial telephone", "diaper",
    "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher",
    "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum",
    "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine",
    "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck",
    "fire screen", "flagpole", "flute", "folding chair", "football helmet",
    "forklift", "fountain", "fountain pen", "four-poster bed", "freight car",
    "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball",
    "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse",
    "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray",
    "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester",
    "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt",
    "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod",
    "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle",
    "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle",
    "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener",
    "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick",
    "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill",
    "magnetic compass", "messenger bag", "mailbox", "tights",
    "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet",
    "megalith", "microphone", "microwave oven", "military uniform", "milk can",
    "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "ford model t", "modem", "monastery", "monitor", "moped",
    "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle",
    "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer",
    "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ",
    "oscilloscope", "overskirt", "bullock cart", "oxygen mask",
    "product packet / packaging", "paddle", "paddle wheel", "padlock",
    "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute",
    "parallel bars", "park bench", "parking meter", "railroad car", "patio",
    "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume",
    "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence",
    "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium",
    "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole",
    "police van", "poncho", "pool table", "soda bottle", "plant pot",
    "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile",
    "projector", "hockey puck", "punching bag", "purse", "quill", "quilt",
    "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair",
    "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker",
    "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone",
    "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield",
    "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart",
    "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask",
    "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel",
    "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar",
    "space heater", "space shuttle", "spatula", "motorboat", "spider web",
    "spindle", "sports car", "spotlight", "stage", "steam locomotive",
    "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall",
    "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen",
    "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing",
    "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot",
    "teddy bear", "television", "tennis ball", "thatched roof", "front curtain",
    "thimble", "threshing machine", "throne", "tile roof", "toaster",
    "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran",
    "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile",
    "typewriter keyboard", "umbrella", "unicycle", "upright piano",
    "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric",
    "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron",
    "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug",
    "whistle", "hair wig", "window screen", "window shade", "Windsor tie",
    "wine bottle", "airplane wing", "wok", "wooden spoon", "wool",
    "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book",
    "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle",
    "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes",
    "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper",
    "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon",
    "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)",
    "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf",
    "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog",
    "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore",
    "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper",
    "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric",
    "gyromitra", "stinkhorn mushroom", "earth star fungus",
    "hen of the woods mushroom", "bolete", "corn cob", "toilet paper",
]


def build_class_prototypes(model, class_names, templates, device, text_batch_size=512,
                           cache_path=None):
    """Build zero-shot class prototype embeddings.

    Encodes all (class × template) texts in large flat batches — far fewer
    forward passes than the old per-class loop (e.g. 156 vs 3000 for 1000
    classes × 80 templates with batch_size=512).

    Optionally caches the result to disk so repeated eval runs skip Llama.

    Args:
        text_batch_size: texts per forward pass (larger = faster, ~512 is safe).
        cache_path: if given, save/load prototypes from this .pt file.

    Returns:
        class_prototypes: Tensor of shape [embedding_dim, n_classes]
    """
    # ── Cache hit ────────────────────────────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached class prototypes from {cache_path}")
        return torch.load(cache_path, map_location=device, weights_only=True)

    model.eval()
    n_classes   = len(class_names)
    n_templates = len(templates)
    total_texts = n_classes * n_templates

    print(f"Encoding {n_classes} classes × {n_templates} templates = "
          f"{total_texts} texts  (batch_size={text_batch_size}, "
          f"{(total_texts + text_batch_size - 1) // text_batch_size} passes) ...")

    # Build flat list: [cls0_tmpl0, cls0_tmpl1, ..., cls999_tmpl79]
    all_texts = [
        template.format(class_name)
        for class_name in class_names
        for template in templates
    ]

    all_embs = []
    amp = device.type == "cuda"
    with torch.no_grad():
        for i in tqdm(range(0, total_texts, text_batch_size), desc="Encoding texts"):
            batch = all_texts[i: i + text_batch_size]
            with torch.autocast(device.type, enabled=amp):
                emb = model.encode_text(batch)      # [B, dim], already L2-normed
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            all_embs.append(emb.cpu().float())

    all_embs = torch.cat(all_embs, dim=0)                      # [N*T, dim]
    all_embs = all_embs.view(n_classes, n_templates, -1)       # [N, T, dim]

    # Normalize → mean → normalize  (matches CLIP-Benchmark exactly)
    all_embs = torch.nn.functional.normalize(all_embs, dim=-1)
    class_embs = all_embs.mean(dim=1)                          # [N, dim]
    class_embs = torch.nn.functional.normalize(class_embs, dim=-1)

    class_prototypes = class_embs.to(device).T                 # [dim, N]

    # ── Cache save ───────────────────────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(class_prototypes.cpu(), cache_path)
        print(f"Cached class prototypes → {cache_path}")

    return class_prototypes


def evaluate(model, class_prototypes, dataloader, device):
    """Evaluate zero-shot top-1 and top-5 accuracy on the given dataloader.

    Images are already transformed tensors from the DataLoader. Passing a
    Tensor to model.encode_image() skips the PIL-specific transform branch,
    so only the frozen vision encoder + vision projector are applied.
    The vision projector already applies L2 normalization.
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    amp = device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating ImageNet-1k"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(device.type, enabled=amp):
                image_features = model.encode_image(images)    # [B, dim], already normalized
                logits = image_features.float() @ class_prototypes.float()   # [B, n_classes]

            # Top-1
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=-1).indices
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Zero-shot ImageNet-1k evaluation for ShareLock")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ShareLock .ckpt checkpoint file")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to base config YAML (default: configs/default_config.yaml)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--text_batch_size", type=int, default=512,
                        help="Texts per Llama forward pass when building class prototypes.")
    parser.add_argument("--prototype_cache", type=str, default=None,
                        help="Path to cache class prototype embeddings (e.g. cache/imagenet_protos.pt). "
                             "Saves Llama encoding time on repeated runs.")
    parser.add_argument("--template", type=str, default=None,
                        help="Single template string, e.g. 'a photo of a {}'. "
                             "If not set, the full 80-template CLIP ensemble is used.")
    parser.add_argument("--imagenet_data_dir", type=str, default="datasets/imagenet-1k",
                        help="Path to locally downloaded ImageNet-1k Parquet files. "
                             "Download first: huggingface-cli download ILSVRC/imagenet-1k "
                             "--repo-type dataset --include 'data/validation*' "
                             "--local-dir datasets/imagenet-1k")
    parser.add_argument("--imagenet_dir", type=str, default=None,
                        help="Path to a local ImageNet validation folder in torchvision format "
                             "(val/n01440764/*.JPEG ...). Overrides --imagenet_data_dir.")
    parser.add_argument("--wordnet_classnames", action="store_true",
                        help="Use raw WordNet labels (with all comma-separated synonyms) instead of "
                             "the default OpenAI/DINOv2 curated class names.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load config & checkpoint ────────────────────────────────────────────
    config = OmegaConf.load(args.config)
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.merge(config, raw_ckpt["hyper_parameters"])

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"  Vision encoder : {config.model.vision_encoder}")
    print(f"  Language encoder: {config.model.language_encoder}")

    model = ShareLock.load_from_checkpoint(
        args.checkpoint, config=config, map_location=device, strict= False
    )
    model = model.to(device)
    model.eval()

    # ── Templates ───────────────────────────────────────────────────────────
    templates = [args.template] if args.template else IMAGENET_TEMPLATES
    print(f"Using {len(templates)} template(s)")

    # ── Load ImageNet-1k validation set ─────────────────────────────────────
    # DINOv2 uses BICUBIC interpolation (not the default BILINEAR in get_transforms())
    image_transforms = tv_transforms.Compose([
        tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.imagenet_dir:
        # ── Local torchvision ImageNet (val/ folder) ─────────────────────────
        # Expects: <imagenet_dir>/<wnid>/*.JPEG  (standard ImageNet structure)
        import torchvision
        # Use torchvision ImageFolder; the folder names are WordNet IDs (wnids)
        tv_dataset = torchvision.datasets.ImageFolder(
            root=args.imagenet_dir,
            transform=image_transforms,
        )
        # Load synset → class name mapping bundled with torchvision
        try:
            meta = torchvision.datasets.ImageNet.META_FILE  # older torchvision
        except AttributeError:
            meta = None

        # Build class names list ordered by torchvision label index
        # tv_dataset.classes are wnids sorted alphabetically; we need human names.
        # Use a bundled JSON mapping if available, else fall back to wnid strings.
        wnid_to_name_path = os.path.join(os.path.dirname(__file__), "imagenet_wnid_to_name.json")
        if os.path.exists(wnid_to_name_path):
            import json
            with open(wnid_to_name_path) as f:
                wnid_to_name = json.load(f)
            class_names = [wnid_to_name.get(w, w) for w in tv_dataset.classes]
        else:
            # Fallback: use torchvision's built-in meta if available
            try:
                import torchvision.datasets.imagenet as _tv_in
                wnid_to_name = {v[0]: v[1][0] for v in _tv_in.IMAGENET_LABELS.values()}
                class_names = [wnid_to_name.get(w, w) for w in tv_dataset.classes]
            except Exception:
                class_names = list(tv_dataset.classes)  # fallback: use wnids

        print(f"Loaded local ImageNet from {args.imagenet_dir}")
        print(f"  {len(class_names)} classes, {len(tv_dataset)} images")

        def tv_collate_fn(batch):
            images = torch.stack([b[0] for b in batch])
            labels = torch.tensor([b[1] for b in batch])
            return {"image": images, "label": labels}

        dataloader = DataLoader(
            tv_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=tv_collate_fn,
            pin_memory=True,
        )

    else:
        # ── Local Parquet files (downloaded via huggingface-cli) ──────────────
        parquet_pattern = os.path.join(args.imagenet_data_dir, "data", "validation-*.parquet")
        parquet_files = sorted(glob_module.glob(parquet_pattern))
        if not parquet_files:
            raise FileNotFoundError(
                f"No validation Parquet files found at {parquet_pattern}\n"
                f"Download first:\n"
                f"  huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset "
                f"--include 'data/validation*' --local-dir {args.imagenet_data_dir}"
            )

        print(f"Loading ImageNet-1k validation from {len(parquet_files)} local Parquet files ...")
        hf_dataset = load_dataset("parquet", data_files=parquet_files, split="train")

        class_names = hf_dataset.features["label"].names
        print(f"  {len(class_names)} classes, {len(hf_dataset)} validation images")

        hf_dataset = hf_dataset.cast_column("image", HFImage(mode="RGB"))

        def hf_transform_fn(batch):
            batch["image"] = [image_transforms(img.convert("RGB")) for img in batch["image"]]
            return batch

        hf_dataset = hf_dataset.with_transform(hf_transform_fn)

        def hf_collate_fn(samples):
            images = torch.stack([torch.as_tensor(s["image"]) for s in samples])
            labels = torch.tensor([s["label"] for s in samples])
            return {"image": images, "label": labels}

        dataloader = DataLoader(
            hf_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=hf_collate_fn,
            pin_memory=True,
        )

    # ── Class names ─────────────────────────────────────────────────────────
    # Default: OpenAI/DINOv2 curated names (single clean name per class).
    # --wordnet_classnames: use raw WordNet labels with all comma-separated synonyms.
    if not args.wordnet_classnames:
        class_names = IMAGENET_OPENAI_CLASSNAMES
        print(f"Using OpenAI/DINOv2 curated class names ({len(class_names)} classes)")
    else:
        print(f"Using raw WordNet class names ({len(class_names)} classes, with synonyms)")

    # ── Build class prototypes ───────────────────────────────────────────────
    # Auto-generate cache path next to checkpoint if not explicitly provided.
    # Cache is keyed to checkpoint so different model runs use separate caches.
    if args.prototype_cache:
        cache_path = args.prototype_cache
    else:
        ckpt_stem = os.path.splitext(args.checkpoint)[0]
        template_tag = "single" if args.template else "80tmpl"
        names_tag = "wordnet" if args.wordnet_classnames else "openai"
        cache_path = f"{ckpt_stem}_protos_{names_tag}_{template_tag}.pt"

    class_prototypes = build_class_prototypes(
        model, class_names, templates, device,
        text_batch_size=args.text_batch_size,
        cache_path=cache_path,
    )
    # class_prototypes: [dim, n_classes]

    # Unload language model to free GPU memory before image evaluation
    if model.language_encoder is not None:
        model.language_encoder.unload_model()
        torch.cuda.empty_cache()

    # ── Run evaluation ───────────────────────────────────────────────────────
    top1, top5 = evaluate(model, class_prototypes, dataloader, device)

    print("\n" + "=" * 50)
    print(f"Zero-shot ImageNet-1k Results")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Templates  : {len(templates)}")
    print(f"  Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5 * 100:.2f}%")
    print("=" * 50)
    print("\nExpected (paper, 80-template ensemble):")
    print("  ShareLock-CC3M  → ~54.5% top-1")
    print("  ShareLock-CC12M → ~62.0% top-1")


if __name__ == "__main__":
    main()
