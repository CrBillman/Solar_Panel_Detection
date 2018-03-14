from argparse import ArgumentParser

from DataRetriever import DataRetriever

def parse_args():
    arg_parser = ArgumentParser()
    
    arg_parser.add_argument('--data_dir', default='../data')
    arg_parser.add_argument('--collection_id', default=3255643,
                            type=int)
    
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    data_retriever = DataRetriever(args.data_dir)
    data_retriever.download_collection_files(args.collection_id)


if __name__ == '__main__':
    main()
