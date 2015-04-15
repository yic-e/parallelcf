#ifndef __RATING_H__
#define __RATING_H__
#include <tuple>
#include <vector>
class ratings {
public:
    ratings();
    void add_rating(int user, int movie, double rating);
    
private:
    std::vector<std::tuple<int, int, double> > __ratings;
    int __user_num;
    int __movie_num;
};

#endif
