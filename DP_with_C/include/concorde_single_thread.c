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
 * @brief Point array destructor.
 * 
 * @param target PointArray to free memory and destruct.
 */
void PA_del(PointArray target);


/**
 * @brief Adjacent matrix initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated ConcordeAdjMatrix instance. 
 */
ConcordeAdjMatrix ADM_init(int num_nodes);


/**
 * @brief Adjacent matrix destructor.
 * 
 * @param target Adjacent matrix pointer to free memory and destruct.
 */
void ADM_del(ConcordeAdjMatrix target);


/**
 * @brief Concorde container initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated ConcordeContainer instance. 
 */
ConcordeContainer Concorde_init(int num_nodes);


/**
 * @brief Concorde container destructor.
 * 
 * @param target Container pointer to free memory and desturct.
 */
void Concorde_del(ConcordeContainer target);

/**
 * @brief TSP solver
 * 
 * @param container  Container pointer which will solve TSP
 * @param file_name  File path to the data
 */
void TSP_solve(ConcordeContainer container, char *file_name);


/**
 * @brief Internal solver with dynamic programming
 * 
 * @param container Container pointer which will solve TSP
 */
void TSP_solve_internal(ConcordeContainer container);