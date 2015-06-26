#ifndef PCT_FUNCTIONS_H
#define PCT_FUNCTIONS_H

// Execution Control Functions
bool is_bad_angle( const int );	// Just for use with Micah's simultated data
void timer( bool, clock_t, clock_t);
void pause_execution();
void exit_program_if( bool );

// Memory transfers and allocations/deallocations
void initial_processing_memory_clean();
void resize_vectors( unsigned int );
void shrink_vectors( unsigned int );
void allocations( const unsigned int );
void reallocations( const unsigned int );
void post_cut_memory_clean();

// Image Initialization/Construction Functions
template<typename T> void initialize_host_image( T*& );
template<typename T> void add_ellipse( T*&, int, double, double, double, double, T );
template<typename T> void add_circle( T*&, int, double, double, double, T );
template<typename O> void import_image( O*&, char* );

// Preprocessing setup and initializations
void write_run_settings();
void assign_SSD_positions();
void initializations();
void count_histories();
void count_histories_old();
void count_histories_v0();
void count_histories_v1();
void reserve_vector_capacity();

// Preprocessing functions
void read_energy_responses( const int, const int, const int );
void combine_data_sets();
void read_data_chunk( const int, const int, const int );
void read_data_chunk_old( const int, const int, const int );
void read_data_chunk_v0( const int, const int, const int );
void read_data_chunk_v02( const int, const int, const int );
void read_data_chunk_v1( const int, const int, const int );
void apply_tuv_shifts( unsigned int );
void convert_mm_2_cm( unsigned int );
void recon_volume_intersections( const int );
void binning( const int );
void calculate_means();
void initialize_stddev();
void sum_squared_deviations( const int, const int );
void calculate_standard_deviations();
void statistical_cuts( const int, const int );
void initialize_sinogram();
void construct_sinogram();
void FBP();
void FBP_image_2_hull();
void filter();
void backprojection();

// Hull-Detection
void hull_initializations();
template<typename T> void initialize_hull( T*&, T*& );
void hull_detection( const int );
void hull_detection_finish();
void SC( const int );
void MSC( const int );
void MSC_edge_detection();
void SM( const int );
void SM_edge_detection();
void SM_edge_detection_2();
void hull_selection();
template<typename T, typename T2> void averaging_filter( T*&, T2*&, int, bool, double );
template<typename H, typename D> void median_filter( H*&, D*&, unsigned int );

// MLP: IN DEVELOPMENT
void create_MLP_test_image();
void MLP();
template<typename O> bool find_MLP_endpoints( O*&, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);
void collect_MLP_endpoints();
unsigned int find_MLP_path( unsigned int*&, double*&, double, double, double, double, double, double, double, double, double, double, int, int, int );
double mean_chord_length( double, double, double, double, double, double );
double mean_chord_length2( double, double, double, double, double, double, double, double );
double EffectiveChordLength(double, double);


// Image Reconstruction
void define_initial_iterate();
void create_hull_image_hybrid();
void image_reconstruction();
template< typename T, typename L, typename R> T discrete_dot_product( L*&, R*&, unsigned int*&, unsigned int );
template< typename A, typename X> double update_vector_multiplier( double, A*&, X*&, unsigned int*&, unsigned int );
template< typename A, typename X> void update_iterate( double, A*&, X*&, unsigned int*&, unsigned int );
// uses mean chord length for each element of ai instead of individual chord lengths
template< typename T, typename RHS> T scalar_dot_product( double, RHS*&, unsigned int*&, unsigned int );
double scalar_dot_product2( double, float*&, unsigned int*&, unsigned int );
template< typename X> double update_vector_multiplier2( double, double, X*&, unsigned int*&, unsigned int );
double update_vector_multiplier22( double, double, float*&, unsigned int*&, unsigned int );
template< typename X> void update_iterate2( double, double, X*&, unsigned int*&, unsigned int );
void update_iterate22( double, double, float*&, unsigned int*&, unsigned int );
template<typename X, typename U> void calculate_update( double, double, X*&, U*&, unsigned int*&, unsigned int );
template<typename X, typename U> void update_iterate3( X*&, U*& );
void generate_error_for_path(double *&,unsigned int,double,double);
double randn(double, double);
void DROP_blocks_error_red_sp(unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*& );
void DROP_blocks_error(unsigned int*&, float*&, float*&, double, unsigned int, double, double*&, unsigned int*& );
void DROP_blocks_error_red( unsigned int*& , float*& ,float*& , double , unsigned int , double , double*& , unsigned int*& );
//void DROP_update_error(float*&, double*&, unsigned int*& , double*&);
void DROP_blocks_red( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*& );
void DROP_blocks( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*& );
void DROP_update( float*&, double*&, unsigned int*& );
void DROP_blocks2( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*&, unsigned int&, unsigned int*& );
void DROP_update2( float*&, double*&, unsigned int*&, unsigned int&, unsigned int*& );
void DROP_blocks3( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*&, unsigned int& );
void DROP_update3( float*&, double*&, unsigned int*&, unsigned int&);
void DROP_blocks_robust2( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*&, double*& );
void DROP_update_robust2( float*&, double*&, unsigned int*&, double*& );



// Write arrays/vectors to file(s)
void binary_2_ASCII();
template<typename T> void array_2_disk( char*, const char*, const char*, T*, const int, const int, const int, const int, const bool );
template<typename T> void vector_2_disk( char*, const char*, const char*, std::vector<T>, const int, const int, const int, const bool );
template<typename T> void t_bins_2_disk( FILE*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const int );
template<typename T> void bins_2_disk( const char*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
template<typename T> void t_bins_2_disk( FILE*, int*&, T*&, const unsigned int, const BIN_ANALYSIS_TYPE, const BIN_ORGANIZATION, int );
template<typename T> void bins_2_disk( const char*, int*&, T*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
FILE* create_MLP_path_file( char* );
template<typename T> void path_data_2_disk(char*, FILE*, unsigned int, unsigned int*, T*&, bool );
void write_MLP_endpoints();
unsigned int read_MLP_endpoints();
void write_MLP_path( FILE*, unsigned int*&, unsigned int);
unsigned int read_MLP_path(FILE*, unsigned int*&);
void read_MLP_path_error(FILE*, float*&, unsigned int);
void export_hull();
void import_hull();
void write_reconstruction_settings();


// Image position/voxel calculation functions
int calculate_voxel( double, double, double );
int position_2_voxel( double, double, double );
int positions_2_voxels(const double, const double, const double, int&, int&, int& );
void voxel_2_3D_voxels( int, int&, int&, int& );
double voxel_2_position( int, double, int, int );
void voxel_2_positions( int, double&, double&, double& );
double voxel_2_radius_squared( int );

// Voxel walk algorithm functions
double distance_remaining( double, double, int, int, double, int );
double edge_coordinate( double, int, double, int, int );
double path_projection( double, double, double, int, double, int, int );
double corresponding_coordinate( double, double, double, double );
void take_2D_step( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );
void take_3D_step( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );

// Host helper functions
template< typename T, typename T2> T max_n( int, T2, ...);
template< typename T, typename T2> T min_n( int, T2, ...);
template<typename T> T* sequential_numbers( int, int );
void bin_2_indexes( int, int&, int&, int& );
inline const char * const bool_2_string( bool b ){ return b ? "true" : "false"; }

// New routine test functions
void command_line_settings( unsigned int, char** );
void read_configurations();
void generate_history_sequence(ULL, ULL, ULL* );
void verify_history_sequence(ULL, ULL, ULL* );
void define_switchmap();
void set_parameter( struct generic_input_container );
void read_parameters();
struct generic_input_container read_parameter( FILE* );
void parameters_2_GPU();
void test_func();
void test_func2( std::vector<int>&, std::vector<double>&);

/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************** Device (GPU) function declarations *****************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

// Preprocessing routines
__device__ bool calculate_intercepts( double, double, double, double&, double& );
__global__ void recon_volume_intersections_GPU( int, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void binning_GPU( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void calculate_means_GPU( int*, float*, float*, float* );
__global__ void sum_squared_deviations_GPU( int, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*  );
__global__ void calculate_standard_deviations_GPU( int*, float*, float*, float* );
__global__ void statistical_cuts_GPU( int, int*, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, bool* );
__global__ void construct_sinogram_GPU( int*, float* );
__global__ void filter_GPU( float*, float* );
__global__ void backprojection_GPU( float*, float* );
__global__ void FBP_image_2_hull_GPU( float*, bool* );

// Hull-Detection
template<typename T> __global__ void initialize_hull_GPU( T* );
__global__ void SC_GPU( const int, bool*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_GPU( const int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void SM_GPU( const int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_edge_detection_GPU( int* );
__global__ void SM_edge_detection_GPU( int*, int* );
__global__ void SM_edge_detection_GPU_2( int*, int* );
__global__ void carve_differences( int*, int* );
//template<typename H, typename D> __global__ void averaging_filter_GPU( H*, D*, int, bool, double );
template<typename D> __global__ void averaging_filter_GPU( D*, D*, int, bool, double );
template<typename D> __global__ void apply_averaging_filter_GPU( D*, D* );

//*********************************************************************MLP & Reconstruction GPU: IN DEVELOPMENT*********************************************************************************************
// MLP & Reconstruction GPU: IN DEVELOPMENT
void generate_trig_tables();
void import_trig_tables();
void generate_scattering_coefficient_table();
void import_scattering_coefficient_table();
void generate_polynomial_tables();
void import_polynomial_tables();
void tables_2_GPU();

void image_reconstruction_GPU_tabulated();//*
template<typename O> __device__ bool find_MLP_endpoints_GPU( O*, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);                                            			//*
__device__ void find_MLP_path_GPU(float*, double, unsigned int, double, double, double, double, double, double, double, double, double, double, double, float, unsigned int*, int&, double&, double&, double& );                        		//*
__device__ void find_MLP_path_GPU_tabulated(float*, double, unsigned int, double, double, double, double, double, double, double, double, double, double, float, unsigned int*, float&, int&, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void collect_MLP_endpoints_GPU(bool*, unsigned int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* , int, int );                               	//*
__global__ void block_update_GPU(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, unsigned int*, float*, unsigned int*, int, int, float);                                          	//*
__global__ void block_update_GPU_tabulated(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, unsigned int*, float*, unsigned int*, int, int, float, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void image_update_GPU(float*, double*, unsigned int*);
void gpu_memory_allocation( const int );//*
void transfer_host_to_device( const int );
void transfer_intermediate_results_device_to_host( const int );
void transfer_intermediate_results_host_to_device( const int );
void transfer_image_device_to_host();//*
void transfer_reconstruction_images();
__global__ void init_image_GPU(float*, unsigned int*);
void reconstruction_cuts (const int, const int);
void reconstruction_cuts_allocations();
void reconstruction_cuts_preallocated(const int, const int);
void drop_cuts (const int , const int );//*															//*
void drop_cuts_allocations(const int);
void drop_cuts_preallocated(const int, const int);
void drop_cuts_memory_clean();
void image_reconstruction_GPU();                                                                                                                                                                   				//*
__device__ double EffectiveChordLength_GPU(double, double);                                                                                                                                              			//*
//**********************************************************************************************************************************************************************************************************
//bool* intersected_hull, unsigned int* first_MLP_voxel, bool* x_hull, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle,
//					 float* xz_exit_angle, int post_cut_protons
//collect_MLP_endpoints_GPU(x, x_hull, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_exit, y_exit, z_exit,  xy_exit_angle,
//					  xz_exit_angle, WEPL, lambda, i, post_cut_protons, a_i, update_value_history, num_intersections_historty);

//__device__ void MLP_GPU();

// Image Reconstruction GPU
//__global__ void image_reconstruction();



// Image Reconstruction
__global__ void create_hull_image_hybrid_GPU( bool*&, float*& );
//template< typename X> __device__ double update_vector_multiplier2( double, double, X*&, int*, int );
__device__ double scalar_dot_product_GPU_2( double, float*&, int*, int );
__device__ double update_vector_multiplier_GPU_22( double, double, float*&, int*, int );
//template< typename X> __device__ void update_iterate2( double, double, X*&, int*, int );
__device__ void update_iterate_GPU_22( double, double, float*&, int*, int );
void update_x();
__global__ void update_x_GPU( float*&, double*&, unsigned int*& );

// Image position/voxel calculation functions
__device__ int calculate_voxel_GPU( double, double, double );
__device__ int positions_2_voxels_GPU(const double, const double, const double, int&, int&, int& );
__device__ int position_2_voxel_GPU( double, double, double );
__device__ void voxel_2_3D_voxels_GPU( int, int&, int&, int& );
__device__ double voxel_2_position_GPU( int, double, int, int );
__device__ void voxel_2_positions_GPU( int, double&, double&, double& );
__device__ double voxel_2_radius_squared_GPU( int );

// Voxel walk algorithm functions
__device__ double distance_remaining_GPU( double, double, int, int, double, int );
__device__ double edge_coordinate_GPU( double, int, double, int, int );
__device__ double path_projection_GPU( double, double, double, int, double, int, int );
__device__ double corresponding_coordinate_GPU( double, double, double, double );
__device__ void take_2D_step_GPU( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );
__device__ void take_3D_step_GPU( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );

// Device helper functions

// New routine test functions
__global__ void test_func_GPU( int* );
__global__ void test_func_device( double*, double*, double* );


#endif /* PCT_FUNCTIONS_H */
