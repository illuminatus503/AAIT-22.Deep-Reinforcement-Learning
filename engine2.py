import argparse
from src.game_player import DiscreteGamePlatform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--num-games", type=int, default=150)
    parser.add_argument("--avg-sample", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=10)
    args = parser.parse_args()

    # Loading the game
    env_id = args.env
    load_checkpoint = args.checkpoint

    game = DiscreteGamePlatform(env_id, load_checkpoint)

    if not load_checkpoint:
        num_games = args.num_games
        game_sample = args.avg_sample
        save_period = args.save_period

        print(f"Begin train for num_games={num_games}")
        game.train_agent(num_games, game_sample, save_period)
    else:
        print("Loading checkpoint...")

    # Record a video
    test_score = game.record_play()
    print(f"**TEST SCORE: {test_score:1.3f}")

    game.close()


if __name__ == "__main__":
    main()
