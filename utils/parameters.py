


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/fed_avg_mini.yaml", type=str)
    parser.add_argument("--exp_name", default="central1", type=str, help="exp name")
    parser.add_argument(
        "--non_iidness", default=1, type=int, help="non-iid degree of distributed data"
    )
    parser.add_argument("--tao_ratio", type=float, default= 1/420, choices=[0.5, 1, 2, 4])
    # optional params
    parser.add_argument("--seed", default=1, type=int, help="using fixed random seed")
    parser.add_argument("--work_dir", default="./runs_mini", type=str, help="output dir")
    # unused params
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    return args