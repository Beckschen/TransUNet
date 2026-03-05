import argparse
import synapseclient


def main(args: argparse.Namespace):
    syn = synapseclient.Synapse()
    syn.login(args.username, args.password)

    entity = syn.get(entity=args.entity, downloadLocation=args.download_dir)
    print("File downloaded successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("username", type=str, help="Username for your Synapse account.")
    parser.add_argument("password", type=str, help="Password to the associated username.")
    parser.add_argument("entity", type=str,
                        help="The SynapseId of the dataset entity. For the Abdomen dataset of the 'Multi-Atlas "
                             "Labeling Beyond the Cranial Vault' challenge, the ids are:\n"
                             "Abdomen: syn3553734\n"
                             "RawData: syn3379050\n"
                             "Reg-Training-Testing: syn3380218\n"
                             "Reg-Training-Training: syn3380229")
    parser.add_argument("download_dir", type=str, help="The location where the file should be downloaded to.")
    main(parser.parse_args())
