#include <stdlib.h>

#include "concorde_single_thread.h"

/**
 * @brief Point Array Initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated PointArray instance.
 */
PointArray PA_init(int num_nodes) {

    PointArray ret = (PointArray)malloc(sizeof(_PointArray));

    ret->points = (Point)malloc(sizeof(_Point)*num_nodes);

    for(int i=0; i<num_nodes; i++) {
        (ret->points[i]).x = -1.0;
        (ret->points[i]).y = -1.0;
    }

    ret->num = num_nodes;
    
    return ret;
}


/**
 * @brief Point Array Destructor.
 * 
 * @param target PointArray to free memory and destruct.
 */
void PA_del(PointArray target);


/**
 * @brief Adjacent Matrix Initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated ConcordeAdjMatrix instance. 
 */
ConcordeAdjMatrix ADM_init(int num_nodes);


/**
 * @brief Adjacent Matrix Destructor.
 * 
 * @param target Adjacent matrix pointer to free memory and destruct.
 */
void ADM_del(ConcordeAdjMatrix target);


/**
 * @brief Concorde Container Initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated ConcordeContainer instance. 
 */
ConcordeContainer Concorde_init(int num_nodes);