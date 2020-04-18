from math import sqrt


class Recommendations:
    def __init__(self, prefs):
        self.prefs = prefs

    # Euclidean Distance Score
    @staticmethod
    def sim_distance(prefs, p1, p2):
        # Get the list of shared_items
        si = {}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item] = 1
        # If they have no ratings in common, return 0
        if len(si) == 0:
            return 0
        # Add up the squares of all the differences
        sum_of_squares = sum([pow(prefs[p1][item] - prefs[p2][item], 2) for item in
                              prefs[p1] if item in prefs[p2]])
        return 1 / (1 + sqrt(sum_of_squares))

    # Returns the Pearson correlation coefficient für p1 & p2
    @staticmethod
    def sim_pearson(prefs, p1, p2):
        si = {}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item] = 1
        n = len(si)
        # Sums of all the preferences
        sum1 = sum([prefs[p1][it] for it in si])
        sum2 = sum([prefs[p2][it] for it in si])
        # Sums of the squares
        sum1_sq = sum([pow(prefs[p1][it], 2) for it in si])
        sum2_sq = sum([pow(prefs[p2][it], 2) for it in si])
        # Sum of the products
        p_sum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
        # Calculate r (Pearson score)
        num = p_sum - sum1 * sum2 / n
        den = sqrt((sum1_sq - pow(sum1, 2) / n) * (sum2_sq - pow(sum2, 2) / n))
        if den == 0:
            return 0
        r = num / den
        return r

    @staticmethod
    def top_matches(prefs, person, similarity, n=5):
        scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
        # sort the list so the hightest score appear at the top
        scores.sort()
        scores.reverse()
        return scores[0:n]

    @staticmethod
    def get_recommendations(prefs, person, similarity):
        totals = {}
        sim_sums = {}
        for other in prefs:
            if other == person:
                continue
            sim = similarity(prefs, person, other)

            # ignore scores of zero or lower
            if sim <= 0:
                continue
            for item in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item]*sim
                sim_sums.setdefault(item, 0)
                sim_sums[item] += sim
                print(sim_sums[item])

        rankings = [(total/sim_sums[item], item) for item, total in totals.items()]
        rankings.sort()
        rankings.reverse()
        return rankings[0][1]




