"""
demo.py — Interactive Terminal Chat Demo

Run this for a quick terminal-based chat session.
No browser needed — great for first-time exploration.

Usage:
    python demo.py
    python demo.py --debug    # show NLP internals on each turn
"""

import sys
import os

# ── Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from chatbot import NLPChatbot


# ── ANSI colors ─────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    RED    = "\033[91m"
    WHITE  = "\033[97m"
    BG_DARK= "\033[40m"


def print_banner():
    print(f"""
{C.CYAN}{C.BOLD}
  ██████╗ ██╗   ██╗██████╗  ██████╗ ████████╗
  ██╔══██╗╚██╗ ██╔╝██╔══██╗██╔═══██╗╚══██╔══╝
  ██████╔╝ ╚████╔╝ ██████╔╝██║   ██║   ██║   
  ██╔═══╝   ╚██╔╝  ██╔══██╗██║   ██║   ██║   
  ██║        ██║   ██████╔╝╚██████╔╝   ██║   
  ╚═╝        ╚═╝   ╚═════╝  ╚═════╝    ╚═╝   
{C.RESET}
{C.DIM}  NLP Chatbot · TF-IDF + Cosine Similarity · Pure Python{C.RESET}
""")


def print_help():
    print(f"""
{C.YELLOW}── Commands ─────────────────────────────────{C.RESET}
  {C.BOLD}debug{C.RESET}     Toggle debug mode (show NLP internals)
  {C.BOLD}stats{C.RESET}     Show session statistics
  {C.BOLD}history{C.RESET}   Show conversation history
  {C.BOLD}reset{C.RESET}     Clear conversation history
  {C.BOLD}help{C.RESET}      Show this help message
  {C.BOLD}quit{C.RESET}      Exit the chatbot
{C.YELLOW}─────────────────────────────────────────────{C.RESET}
""")


def print_debug_info(debug: dict):
    print(f"\n  {C.DIM}┌─ NLP Debug ────────────────────────────────────────{C.RESET}")
    print(f"  {C.DIM}│ Tokens:     {debug['tokens']}{C.RESET}")
    print(f"  {C.DIM}│ Stemmed:    {debug['stemmed_tokens']}{C.RESET}")
    print(f"  {C.DIM}│ Processed:  {debug['preprocessed']}{C.RESET}")
    print(f"  {C.DIM}│ Best match: \"{debug['best_pattern']}\"{C.RESET}")
    if debug.get("entities"):
        print(f"  {C.DIM}│ Entities:   {debug['entities']}{C.RESET}")
    print(f"  {C.DIM}│ Top intents:{C.RESET}")
    for tag, score in debug["top_intents"][:4]:
        bar = "█" * int(score * 20)
        color = C.GREEN if score == debug["top_intents"][0][1] else C.DIM
        print(f"  {C.DIM}│   {color}{tag:20s}{C.RESET} {C.DIM}{score:.4f} {bar}{C.RESET}")
    print(f"  {C.DIM}└────────────────────────────────────────────────────{C.RESET}\n")


def print_stats(stats: dict):
    print(f"\n{C.YELLOW}── Session Stats ────────────────────────────{C.RESET}")
    for key, val in stats.items():
        if key == "intent_distribution":
            print(f"  {C.BOLD}Intent distribution:{C.RESET}")
            for intent, count in val.items():
                print(f"    {intent}: {count}")
        else:
            label = key.replace("_", " ").title()
            print(f"  {C.BOLD}{label}:{C.RESET} {val}")
    print(f"{C.YELLOW}─────────────────────────────────────────────{C.RESET}\n")


def confidence_color(score: float) -> str:
    if score >= 0.5:
        return C.GREEN
    elif score >= 0.25:
        return C.YELLOW
    else:
        return C.RED


def run():
    debug_mode = "--debug" in sys.argv

    print_banner()

    bot = NLPChatbot("intents.json")

    print(f"\n{C.GREEN}● Chatbot ready!{C.RESET} Type {C.BOLD}help{C.RESET} for commands, {C.BOLD}quit{C.RESET} to exit.\n")

    if debug_mode:
        print(f"{C.YELLOW}⚙  Debug mode ON{C.RESET}\n")

    while True:
        try:
            # Prompt
            user_input = input(f"{C.CYAN}{C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Goodbye!{C.RESET}\n")
            break

        if not user_input:
            continue

        # Built-in commands
        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            print(f"\n{C.DIM}Goodbye! Thanks for chatting.{C.RESET}\n")
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "debug":
            debug_mode = not debug_mode
            state = f"{C.GREEN}ON{C.RESET}" if debug_mode else f"{C.RED}OFF{C.RESET}"
            print(f"\n  ⚙  Debug mode {state}\n")
            continue
        elif cmd == "stats":
            print_stats(bot.get_stats())
            continue
        elif cmd == "history":
            history = bot.conversation_history
            if not history:
                print(f"\n  {C.DIM}No history yet.{C.RESET}\n")
            else:
                print(f"\n{C.YELLOW}── Conversation History ─────────────────────{C.RESET}")
                for h in history:
                    print(f"  [{h['turn']}] {C.CYAN}You:{C.RESET} {h['user']}")
                    print(f"       {C.MAGENTA}Bot:{C.RESET} {h['bot']}")
                    print(f"       {C.DIM}Intent: {h['intent']} | Conf: {h['confidence']:.4f}{C.RESET}")
                print(f"{C.YELLOW}─────────────────────────────────────────────{C.RESET}\n")
            continue
        elif cmd == "reset":
            bot.reset()
            print(f"\n  ✅ Conversation reset.\n")
            continue

        # Chat
        result = bot.respond(user_input, debug=debug_mode)

        score = result["confidence"]
        score_color = confidence_color(score)

        print(f"\n{C.MAGENTA}{C.BOLD}Bot:{C.RESET} {result['response']}")
        print(f"    {C.DIM}[{result['tag']} · {score_color}{score:.4f}{C.RESET}{C.DIM} · {result['time_ms']}ms]{C.RESET}\n")

        if debug_mode and "debug" in result:
            print_debug_info(result["debug"])


if __name__ == "__main__":
    run()
