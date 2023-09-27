/**
 * @file concorde_single_thread.h
 * @author Chan-young Lee (ckckdud123@gmail.com)
 * @brief TSP solver with dynamic programming (single-threaded version)
 * @version 1.0
 * @date 13 Sep 2023
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <stdlib.h>
#include <math.h>

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
        // -1.0 Indicates that points are null.
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
void PA_del(PointArray target) {

    free(target->points);
    free(target);
}


/**
 * @brief Adjacent matrix initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated ConcordeAdjMatrix instance. 
 */
ConcordeAdjMatrix ADM_init(int num_nodes) {

    ConcordeAdjMatrix ret = (ConcordeAdjMatrix)malloc(sizeof(_ConcordeAdjMatrix));
    
    ret->num = num_nodes;

    ret->matrix = (double **)malloc(sizeof(double *) * num_nodes);

    for(int i=0; i<num_nodes; i++) {
        ret->matrix[i] = (double *)malloc(sizeof(double) * num_nodes);
        // -1.0 Indicates null.
        for(int j=0; j<num_nodes; j++) ret->matrix[i][j] = -1.0;
    }

    return ret;
}


/**
 * @brief Adjacent matrix destructor.
 * 
 * @param target Adjacent matrix pointer to free memory and destruct.
 */
void ADM_del(ConcordeAdjMatrix target) {

    for(int i=0; i<target->num; i++) {
        free(target->matrix[i]);
    }
    free(target->matrix);
    free(target);
}


/**
 * @brief Concorde container initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated ConcordeContainer instance. 
 */
ConcordeContainer Concorde_init(int num_nodes) {

    ConcordeContainer ret = (ConcordeContainer)malloc(sizeof(_ConcordeContainer));

    ret->adm = ADM_init(num_nodes);
    ret->parr = PA_init(num_nodes);

    ret->ground_truth = (int *)malloc(sizeof(int) * (num_nodes + 1));
    ret->answer = (int *)malloc(sizeof(int) * (num_nodes + 1));

    ret->memo = (double *)malloc(sizeof(double) * (int)((num_nodes-1) * pow(2.0, (double)(num_nodes-1))));
    for(int i=0; i<(int)((num_nodes-1) * pow(2.0, (double)(num_nodes-1))); i++) {
        ret->memo[i] = -1.0;
    }

    ret->num = num_nodes;

    return ret;
}


/**
 * @brief Concorde container destructor.
 * 
 * @param target Container pointer to free memory and desturct.
 */
void Concorde_del(ConcordeContainer target) {

    ADM_del(target->adm);
    PA_del(target->parr);
    free(target->ground_truth);
    free(target->answer);
    free(target);
}

/**
 * @brief TSP solver
 * 
 * @param container  Container pointer which will solve TSP with dynamic programming
 * @param file_name  File path to the data
 */
void TSP_solve_DP(ConcordeContainer container, char *file_name) {
    
}


/**
 * @brief Internal solver with dynamic programming
 * 
 * @param container Container pointer which will solve TSP
 * @param depth Indicates current depth of subproblem
 * @param index Indicates currently solving index
 * @param mask Indicates visited nodes
 */
double TSP_solve_DP_internal(ConcordeContainer container, int depth, int index, int mask) {
    
    if(depth == 2) {
        return container->adm->matrix[0][index];
    }

    
}