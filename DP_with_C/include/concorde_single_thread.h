#ifndef __CONCORDE_GRAPH__
#define __CONCORDE_GRAPH__

// Struct declarations

typedef struct __point {

    double x;
    double y;

} _Point;

typedef _Point *Point;


typedef struct __point_array {

    Point points;
    int num;

} _PointArray;

typedef _PointArray *PointArray;


typedef struct __concorde_adj_matrix {

    double **matrix;
    int num;

} _ConcordeAdjMatrix;

typedef _ConcordeAdjMatrix *ConcordeAdjMatrix;


typedef struct __concorde_container {

    PointArray parr;
    ConcordeAdjMatrix adm;
    
    char buffer[1024];
    int fd;
    int num;
    int *answer;

} _ConcordeContainer;

typedef _ConcordeContainer *ConcordeContainer;

// End of struct declarations


// Method definitions

/**
 * @brief Point array initializer.
 * 
 * @param num_nodes Indicates how many nodes are there.
 * @return Allocated PointArray instance.
 */
PointArray PA_init(int num_nodes);


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

#endif