#include "pCT_data.h"
#include "pCT_functions.h"

/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************ Testing Functions and Functions in Development ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void test_func()
{
	char filename[256];
	char* name = "FBP_med7";
	sprintf( filename, "%s%s/%s%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER, name, ".bin" );
	float* image = (float*)calloc( NUM_VOXELS, sizeof(float));
	import_image( image, filename );
	array_2_disk( name, OUTPUT_DIRECTORY, OUTPUT_FOLDER, image, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//read_config_file();
	//double voxels[4] = {1,2,3,4};
	//std::copy( x_hull_h, x_hull_h + NUM_VOXELS, x_h );
	//std::function<double(int, int)> fn1 = my_divide;                    // function
	//int x = 2;
	//int y = 3;
	//cout << func_pass_test(x,y, fn1) << endl;
	//std::function<double(double, double)> fn2 = my_divide2;                    // function
	//double x2 = 2;
	//double y2 = 3;
	//cout << func_pass_test(x2,y2, fn2) << endl;
	//std::function<int(int)> fn2 = &half;                   // function pointer
	//std::function<int(int)> fn3 = third_t();               // function object
	//std::function<int(int)> fn4 = [](int x){return x/4;};  // lambda expression
	//std::function<int(int)> fn5 = std::negate<int>();      // standard function object
	//create_MLP_test_image();
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//MLP_test();
	//array_2_disk( "MLP_image", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//double* x = (double*) calloc(4, sizeof(double) );
	//double* y = (double*) calloc(4, sizeof(double) );
	//double* z = (double*) calloc(4, sizeof(double) );

	//double* x_d, *y_d, *z_d;
	////sinogram_filtered_h = (float*) calloc( NUM_BINS, sizeof(float) );
	//cudaMalloc((void**) &x_d, 4*sizeof(double));
	//cudaMalloc((void**) &y_d, 4*sizeof(double));
	//cudaMalloc((void**) &z_d, 4*sizeof(double));

	//cudaMemcpy( x_d, x, 4*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy( y_d, y, 4*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy( z_d, z, 4*sizeof(double), cudaMemcpyHostToDevice);

	//dim3 dimBlock( 1 );
	//dim3 dimGrid( 1 );   	
	//test_func_device<<< dimGrid, dimBlock >>>( x_d, y_d, z_d );

	//cudaMemcpy( x, x_d, 4*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy( y, y_d, 4*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy( z, z_d, 4*sizeof(double), cudaMemcpyDeviceToHost);

	//for( unsigned int i = 0; i < 4; i++)
	//{
	//	printf("%3f\n", x[i] );
	//	printf("%3f\n", y[i] );
	//	printf("%3f\n", z[i] );
	//	//cout << x[i] << endl; // -8.0
	//	//cout << y[i] << endl;
	//	//cout << z[i] << endl;
	//}
}
void test_func2( std::vector<int>& bin_numbers, std::vector<double>& data )
{
	int angular_bin = 8;
	int v_bin = 14;
	int bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);

	v_bin = 15;
	bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);

	angular_bin = 30;
	v_bin = 14;
	bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);

	v_bin = 16;
	bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);
	//cout << smallest << endl;
	//cout << min_n<double, int>(9, 1, 2, 3, 4, 5, 6, 7, 8, 100 ) << endl;
	//cout << true << endl;
	//FILE * pFile;
	//char data_filename[MAX_INTERSECTIONS];
	//sprintf(data_filename, "%s%s/%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER, "myfile.txt" );
	//pFile = fopen (data_filename,"w+");
	//int ai[1000];
	//cout << pow(ROWS, 2.0) + pow(COLUMNS,2.0) + pow(SLICES,2.0) << " " <<  sqrt(pow(ROWS, 2.0) + pow(COLUMNS,2.0) + pow(SLICES,2.0)) << " " << max_path_elements << endl;
	////pFile = freopen (data_filename,"a+", pFile);
	//for( unsigned int i = 0; i < 10; i++ )
	//{
	//	//int ai[i];
	//	for( int j = 0; j < 10 - i; j++ )
	//	{
	//		ai[j] = j; 
	//		//cout << ai[i] << endl;
	//	}
	//	write_path(data_filename, pFile, 10-i, ai, false);
	//}
	
	//int myints[] = {16,2,77,29};
	//std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );

	//int x_elements = 5;
	//int y_elements = 10;
	////int x[] = {10, 20,30};
	////int angle_array[];

	//int* x = (int*) calloc( x_elements, sizeof(int));
	//int* y = (int*) calloc( y_elements, sizeof(int));
	//for( unsigned int i = 0; i < x_elements; i++ )
	//{
	//	x[i] = 10*i;
	//}
	//for( unsigned int i = 0; i < y_elements; i++ )
	//{
	//	y[i] = i;
	//}
	////cout << sizeof(&(*x)) << endl;

	//test_va_arg( fifth, BY_BIN, x_elements, x, y_elements, y );
	//else
	//{
	//	//int myints[] = {16,2,77,29};
	//	//std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
	//	va_list specific_bins;
	//	va_start( specific_bins, bin_order );
	//	int* angle_array = va_arg(specific_bins, int* );		
	//	int* v_bins_array = va_arg(specific_bins, int* );
	//	std::vector<int> temp ( angle_array,  angle_array + sizeof(angle_array) / sizeof(int) );
	//	angles = temp;
	//	std::vector<int> temp2 ( v_bins_array,  v_bins_array + sizeof(v_bins_array) / sizeof(int) );
	//	v_bins = temp2;
	//	//angles = va_arg(specific_bins, int* );
	//	//v_bins = va_arg(specific_bins, int* );
	//	va_end(specific_bins);
	//	angular_bins.resize(angles.size());
	//	std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//}
	//char* data_format = INT_FORMAT;
	////int x[] = {10, 20,30};
	////int y[] = {1, 2,3};
	//int* x = (int*) calloc( 3, sizeof(int));
	//int* y = (int*) calloc( 3, sizeof(int));
	//for( unsigned int i = 0; i < 3; i++)
	//{
	//	x[i] = 10*i;
	//	y[i] = i;
	//}
	//for( unsigned int i = 0; i < 3; i++)
	//{
	//	cout << x[i] << " " << y[i] << endl;
	//}

	////int type_var;
	//int* intersections = (int*) calloc( 3, sizeof(int));
	//std::iota( intersections, intersections + 3, 0 );
	//double z = discrete_dot_product<double>(x, y, intersections, 3);
	//printf("%d %d %d\n%f %f %f\n", x[1], y[1], z, x[1], y[1], z);
	//create_MLP_test_image();
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//MLP_test();
	//array_2_disk( "MLP_image", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//int* x = (int*)calloc(10, sizeof(int));
	//int* y = (int*)calloc(10, sizeof(int));
	//std::vector<int*> paths;
	//std::vector<int> num_paths;
	//paths.push_back(x);
	//paths.push_back(y);
	//num_paths.push_back(10);
	//num_paths.push_back(10);

	//std::vector<int> x_vec(10);
	//std::vector<int> y_vec(10);
	//std::vector<std::vector<int>> z_vec;

	//
	//for( int j = 0; j < 10; j++ )
	//{
	//	x[j] = j;
	//	y[j] = 2*j;
	//	x_vec[j] = j;
	//	y_vec[j] = 2*j;
	//}
	//for( unsigned int i = 0; i < paths.size(); i++ )
	//{
	//	for( int j = 0; j < num_paths[i]; j++ )
	//		cout << (paths[i])[j] << endl;
	//}

	//z_vec.push_back(x_vec);
	//z_vec.push_back(y_vec);

	//for( unsigned int i = 0; i < z_vec.size(); i++ )
	//{
	//	for( int j = 0; j < (z_vec[i]).size(); j++)
	//		cout << (z_vec[i])[j] << endl;

	//}

	//std::vector<std::vector<int>> t_vec(5);
	//std::vector<int> temp_vec;
	////temp_vec = new std::vector<int>();
	////std::vector<int> temp_vec = new std::vector<int>(5);
	//for( unsigned int i = 0; i < t_vec.size(); i++ )
	//{
	//	//temp_vec = new std::vector<int>();
	//	//std::vector<int> temp_vec(i);
	//	for( int j = 0; j < i; j++ )
	//	{
	//		temp_vec.push_back(i*j);
	//		//temp_vec[j] = i*j;
	//	}
	//	t_vec[i] = temp_vec;
	//	temp_vec.clear();
	//	//delete temp_vec;
	//}
	//for( unsigned int i = 0; i < t_vec.size(); i++ )
	//{
	//	for( int j = 0; j < t_vec[i].size(); j++ )
	//	{
	//		cout << (t_vec[i])[j] << endl;
	//	}
	//}

	//for( int i = 0, float df = 0.0; i < 10; i++)
	//	cout << "Hello" << endl;
	////int x[] = {2,3,4,6,7};
	////test_func_3();
	//int x[] = {-1, 0, 1};
	//bool y[] = {0,0,0}; 
	//std::transform( x, x + 3, x, y, std::logical_or<int> () );
	//for(unsigned int i = 0; i < 3; i++ )
	//	std::cout << y[i] << std::endl;
	//std::initializer_list<int> mylist;
	//std::cout << sizeof(bool) << sizeof(int) << std::endl;
	//mylist = { 10, 20, 30 };
	////std::array<int,10> y = {1,2,3,4};
	////auto ptr = y.begin();

	//int y[20];
	//int index = 0;
	//for( unsigned int i = 0; i < 20; i++ )
	//	y[index++] = i;
	//for( unsigned int i = 0; i < 20; i++ )
	//	std::cout << y[i] << std::endl;

	//int* il = { 10, 20, 30 };
	//auto p1 = il.begin();
	//auto fn_five = std::bind (my_divide,10,2);               // returns 10/2
  //std::cout << fn_five() << '\n';  

	//std::vector<int> bin_numbers;
	//std::vector<float> WEPLs;
	//test_func2( bin_numbers, WEPLs );
	//int angular_bin = 8;
	//int v_bin = 14;
	//int bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;

	//std::cout << typeid(bin_numbers.size()).name() << std::endl;
	//std::cout << typeid(1).name() << std::endl;
	//printf("%03d %03d\n", bin_numbers.size(), WEPLs.size() );


	///*for( unsigned int i = 0; i < WEPLs.size(); i++ )
	//{
	//	printf("%d %3f\n", bin_numbers[i], WEPLs[i] );
	//}*/
	//char filename[256];
	//FILE* output_file;
	//int angles[] = {32,120};
	//int v_bins[] = {14,15,16};
	//float* sino = (float*) std::calloc( 10, sizeof(float));
	//auto it = std::begin(angles);
	//std::cout << sizeof(&*sino)/sizeof(float) << std::endl << std::endl;
	//std::vector<int> angles_vec(angles, angles + sizeof(angles) / sizeof(int) );
	//std::vector<int> v_bins_vec(v_bins, v_bins + sizeof(v_bins) / sizeof(int) );
	//std::vector<int> angular_bins = angles_vec;
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//int num_angles = sizeof(angles)/sizeof(int);
	//int num_v_bins = sizeof(v_bins)/sizeof(int);
	//std::cout << sizeof(v_bins) << " " << sizeof(angles) << std::endl;
	//std::cout << num_angles << " " << num_v_bins << std::endl;
	//std::cout << angles_vec.size() << " " << angular_bins.size() << std::endl;
	//bins_2_disk( "bin data", bin_numbers, WEPLs, COUNTS, ALL_BINS, BY_HISTORY );
	//bins_2_disk( "bin data", bin_numbers, WEPLs, COUNTS, ALL_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//bins_2_disk( "bin_counts", bin_numbers, WEPLs, COUNTS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//bins_2_disk( "bin_means", bin_numbers, WEPLs, MEANS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//bins_2_disk( "bin_members", bin_numbers, WEPLs, MEMBERS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//for( unsigned int i = 0; i < angular_bins.size(); i++ )
	//	std::cout << angular_bins[i] << std::endl;
	////std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind( std::divides<int>(), 4 ) );

	//
	//auto f1 = std::bind(my_divide, _1, 10);
	////auto triple = std::mem_fn (my_divide, _1);
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(),  f1 );
	//for( unsigned int i = 0; i < angular_bins.size(); i++ )
	//	std::cout << angular_bins[i] << std::endl;
	//int angles[] = {32,120,212};
}
void test_transfer()
{
	unsigned int N_x = 4;
	unsigned int N_y = 4;
	unsigned int N_z = 4;

	double* x = (double*) calloc(N_x, sizeof(double) );
	double* y = (double*) calloc(N_y, sizeof(double) );
	double* z = (double*) calloc(N_z, sizeof(double) );

	double* x_d, *y_d, *z_d;

	cudaMalloc((void**) &x_d, N_x*sizeof(double));
	cudaMalloc((void**) &y_d, N_y*sizeof(double));
	cudaMalloc((void**) &z_d, N_z*sizeof(double));

	cudaMemcpy( x_d, x, N_x*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( y_d, y, N_y*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( z_d, z, N_z*sizeof(double), cudaMemcpyHostToDevice);

	dim3 dimBlock( 1 );
	dim3 dimGrid( 1 );   	
	test_func_device<<< dimGrid, dimBlock >>>( x_d, y_d, z_d );

	cudaMemcpy( x, x_d, N_x*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy( y, y_d, N_y*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy( z, z_d, N_z*sizeof(double), cudaMemcpyDeviceToHost);

	for( unsigned int i = 0; i < N_x; i++)
	{
		printf("%3f\n", x[i] );
		printf("%3f\n", y[i] );
		printf("%3f\n", z[i] );
		//cout << x[i] << endl; // -8.0
		//cout << y[i] << endl;
		//cout << z[i] << endl;
	}
}
void test_transfer_GPU(double* x, double* y, double* z)
{
	//x = 2;
	//y = 3;
	//z = 4;
}
__global__ void test_func_device( double* x, double* y, double* z )
{
	//x = 2;
	//y = 3;
	//z = 4;
}
__global__ void test_func_GPU( int* a)
{
	//int i = threadIdx.x;
	//std::string str;
	double delta_yx = 1.0/1.0;
	double x_to_go = 0.024;
	double y_to_go = 0.015;
	double y_to_go2 = y_to_go;
	double y_move = delta_yx * x_to_go;
	if( -1 )
		printf("-1");
	if( 1 )
		printf("1");
	if( 0 )
		printf("0");
	y_to_go -= !sin(delta_yx)*y_move;

	y_to_go2 -= !sin(delta_yx)*delta_yx * x_to_go;

	printf(" delta_yx = %8f y_move = %8f y_to_go = %8f y_to_go2 = %8f\n", delta_yx, y_move, y_to_go, y_to_go2 );
	double y = 1.36;
	////int voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) / VOXEL_WIDTH );
	//int voxel_y_out = int( ( RECON_CYL_RADIUS - y ) / VOXEL_HEIGHT );
	////int voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
	//double voxel_y_float;
	//double y_inside2 = ((( RECON_CYL_RADIUS - y ) / VOXEL_HEIGHT) - voxel_y_out) * VOXEL_HEIGHT;
	//double y_inside = modf( ( RECON_CYL_RADIUS - y) /VOXEL_HEIGHT, &voxel_y_float)*VOXEL_HEIGHT;
	//printf(" voxel_y_float = %8f voxel_y_out = %d\n", voxel_y_float, voxel_y_out );
	//printf(" y_inside = %8f y_inside2 = %8f\n", y_inside, y_inside2 );
	//printf("Hello %d", i);
	float x = 1.0;
	y = 1.0;
	float z = abs(2.0) / abs( x - y );
	float z2 = abs(-2.0) / abs( x - y );
	float z3 = z*x;
	bool less = z < z2;
	bool less2 = x < z;
	bool less3 = x < z2;
	if( less )
		a[0] = 1;
	if( less2 )
		a[1] = 1;
	if( less3 )
		a[2] = 1;

	printf("%3f %3f %3f %d %d %d\n", z, z2, z3, less, less2, less3);
	//int voxel_x = blockIdx.x;
	//int voxel_y = blockIdx.y;	
	//int voxel_z = threadIdx.x;
	//int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	//int x = 0, y = 0, z = 0;
	//test_func_device( x, y, z );
	//image[voxel] = x * y * z;
}
