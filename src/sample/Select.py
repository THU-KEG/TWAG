import json
import argparse
import os


def work(args):
    if args.animal_topic_num == 5:
        animal_clusts = [
            ['Distribution'],
            ['Taxonomy', 'Species', 'Subspecies', 'Classification'],
            ['Description', 'Habitat', 'Ecology', 'Behaviour', 'Biology', 'Diet', 'Feeding', 'Breeding', 'Reproduction',
             'Life cycle'],
            ['Status', 'Conservation', 'Conservation status'],
            ['NOISE']
        ]
    elif args.animal_topic_num == 3:
        animal_clusts = [
            ['Distribution', 'Taxonomy', 'Species', 'Subspecies', 'Classification',
             'Life cycle', 'Status', 'Conservation', 'Conservation status'],
            ['Description', 'Habitat', 'Ecology', 'Behaviour', 'Biology', 'Diet', 'Feeding', 'Breeding',
             'Reproduction'],
            ['NOISE']
        ]

    animal_path = os.path.join(args.classifier_dir, 'animal', 'TopicList.txt')

    with open(animal_path, 'w') as fin:
        json.dump(animal_clusts, fin, ensure_ascii=False)

    film_clusts = [
        ['Cast', 'Casting'],
        ['Plot', 'Synopsis', 'Plot summary'],
        ['Production', 'Filming', 'Development'],
        ['Reception', 'Critical reception', 'Critical response', 'Awards', 'Accolades', 'Awards and nominations'],
        ['Box office'],
        ['NOISE']
    ]

    film_path = os.path.join(args.classifier_dir, 'film', 'TopicList.txt')
    with open(film_path, 'w') as fin:
        json.dump(film_clusts, fin, ensure_ascii=False)

    # 10 titles
    # company_clusts = [
    #     ['History'],
    #     ['Products', 'Services', 'Destinations'],
    #     ['Fleet', 'Operations'],
    #     ['Awards'],
    #     ['NOISE']
    # ]

    # 20 titles
    company_clusts = [
        ['History', 'Company history'],
        ['Products', 'Services', 'Destinations', 'Products and services', 'Technology'],
        ['Fleet', 'Subsidiaries', 'Operations', 'Locations'],
        ['Awards'],
        ['NOISE']
    ]

    # 30 titles
    # company_clusts = [
    #      ['History', 'Company history', 'Ownership'],
    #      ['Products', 'Services', 'Destinations', 'Products and services', 'Technology'],
    #      ['Fleet', 'Subsidiaries', 'Operations', 'Locations'],
    #      ['Awards', 'Controversies', 'Controversy', 'Criticism', 'Accidents and incidents', 'Reception'],
    #      ['NOISE']
    #  ]

    company_path = os.path.join(args.classifier_dir, 'company', 'TopicList.txt')
    with open(company_path, 'w') as fin:
        json.dump(company_clusts, fin, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal_topic_num', type=int, default=5)
    args = parser.parse_args()
    work(args)
