from recommendation.pydelicious import get_popular, get_userposts, get_urlposts


class Deliciousrec:

    @staticmethod
    def initialize_user_dict(tag, count=5):
        user_dict = {}
        for p1 in get_popular(tag=tag)[0:count]:
            for p2 in get_urlposts(p1['href']):
                user = p2['user']
                user_dict[user] = {}
        return user_dict


