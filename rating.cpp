#include "rating.h"

ratings::ratings():user_num(0),
                   movie_num(0){
    
}

void ratings::add_rating(int user, int movie, double rating){
    __ratings.push_back(std::make_tuple(user, movie, rating));
    if(user >= user_num){
        user_num = user + 1;
    }
    if(movie >= movie_num){
        movie_num = movie + 1;
    }
}

