import recommendation.recommendations as rm


def main():
    critics = {
        'Lisa Rose': {
            'Lady in the Water': 2.5,
            'Snakes on a Plane': 3.5,
            'Just My Luck': 3.0,
            'Superman Returns': 3.5,
            'You, Me and Dupree': 2.5,
            'The Night Listener': 3.0,
        },
        'Gene Seymour': {
            'Lady in the Water': 3.0,
            'Snakes on a Plane': 3.5,
            'Just My Luck': 1.5,
            'Superman Returns': 5.0,
            'The Night Listener': 3.0,
            'You, Me and Dupree': 3.5,
        },
        'Michael Phillips': {
            'Lady in the Water': 2.5,
            'Snakes on a Plane': 3.0,
            'Superman Returns': 3.5,
            'The Night Listener': 4.0,
        },
        'Claudia Puig': {
            'Snakes on a Plane': 3.5,
            'Just My Luck': 3.0,
            'The Night Listener': 4.5,
            'Superman Returns': 4.0,
            'You, Me and Dupree': 2.5,
        },
        'Mick LaSalle': {
            'Lady in the Water': 3.0,
            'Snakes on a Plane': 4.0,
            'Just My Luck': 2.0,
            'Superman Returns': 3.0,
            'The Night Listener': 3.0,
            'You, Me and Dupree': 2.0,
        },
        'Jack Matthews': {
            'Lady in the Water': 3.0,
            'Snakes on a Plane': 4.0,
            'The Night Listener': 3.0,
            'Superman Returns': 5.0,
            'You, Me and Dupree': 3.5,
        },
        'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
                 'Superman Returns': 4.0},
    }
    funs = rm.Recommendations(critics)
    euclidean_distance_score = funs.sim_distance(critics, 'Lisa Rose', 'Gene Seymour')
    print('Euclidean distance sorce is ', euclidean_distance_score)

    pearson_correlation_score = funs.sim_pearson(critics, 'Lisa Rose', 'Gene Seymour')
    print('Pearson correlation score is', pearson_correlation_score)

    top_matches = funs.top_matches(critics, 'Toby', funs.sim_pearson, n=1)
    print('The best match for Toby is', top_matches)

    recommendation = funs.get_recommendations(critics, 'Toby', funs.sim_pearson)
    print('The recommendation for Toby is', recommendation)


if __name__ == '__main__':
    main()
    print('Happy Ending!')
