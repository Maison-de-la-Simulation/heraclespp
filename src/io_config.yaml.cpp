#include "io_config.yaml.hpp"

char const* const io_config = R"IO_CONFIG(
pdi:
  metadata:
    ndim: int
    max_iter: int
    frequency: int
    iter: int
    time_out: double
    current_time: double
    mpi_rank: int
    mpi_size: int
    nx_glob_ng: {type: array, subtype: int, size: 3}
    nx_local_ng: {type: array, subtype: int, size: 3}
    nx_local_wg: {type: array, subtype: int, size: 3}
    n_ghost: {type: array, subtype: int, size: 3}
    start: {type: array, subtype: int, size: 3}
    restart_filename_size: int
    restart_filename: { type: array, subtype: char, size: $restart_filename_size }
    

  data: # this describe the data that will be send by each proc
    u:   { type: array, subtype: double, size: ['$ndim', '$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]'] }
    rho: { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    P:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    E:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    x:   { type: array, subtype: double, size: '$nx_local_wg[0]+1' }
    y:   { type: array, subtype: double, size: '$nx_local_wg[1]+1' }
    z:   { type: array, subtype: double, size: '$nx_local_wg[2]+1' }

  plugins:
    mpi:
    decl_hdf5:
      - file: test_${iter:08}.h5
        collision_policy: write_into
        communicator: '$MPI_COMM_WORLD'
        on_event: write_file
        datasets: # this is the global data size (data in the final h5 fil)
          u:   {type: array, subtype: double, size: ['$ndim', '$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          rho: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]' ] }
          P:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]' ] }
          E:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]' ] }
          x:   {type: array, subtype: double, size: ['$nx_glob_ng[0]+1'] }
          y:   {type: array, subtype: double, size: ['$nx_glob_ng[1]+1'] }
          z:   {type: array, subtype: double, size: ['$nx_glob_ng[2]+1'] }
        write:
          iter:
          current_time:
          u:
            memory_selection:
              size: ['$ndim', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$ndim', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ 0, '$start[2]', '$start[1]', '$start[0]']
          rho:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]'] # size of data to be extracted from the proc
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']            # starting position of data to be extracted from the proc
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]'] # size of data to be put in the final h5 file
              start: [ '$start[2]', '$start[1]', '$start[0]']                 # position of the data in the global index
          P:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ '$start[2]', '$start[1]', '$start[0]']
          E:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ '$start[2]', '$start[1]', '$start[0]']
          x:
            memory_selection:
              size: ['$nx_local_ng[0]+1']
              start: ['$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[0]+1']
              start: ['$start[0]']
          y:
            memory_selection:
              size: ['$nx_local_ng[1]+1']
              start: ['$n_ghost[1]']
            dataset_selection:
              size: ['$nx_local_ng[1]+1']
              start: ['$start[1]']
          z:
            memory_selection:
              size: ['$nx_local_ng[2]+1']
              start: ['$n_ghost[2]']
            dataset_selection:
              size: ['$nx_local_ng[2]+1']
              start: ['$start[2]']

      - file: ${restart_filename}
        on_event: read_file
        communicator: '${MPI_COMM_WORLD}'
        read:
          iter:
          current_time:
          rho:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]'] # size of data to be extracted from the proc
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']            # starting position of data to be extracted from the proc
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]'] # size of data to be put in the final h5 file
              start: [ '$start[2]', '$start[1]', '$start[0]']                 # position of the data in the global index
          u:
            memory_selection:
              size: ['$ndim', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$ndim', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ 0, '$start[2]', '$start[1]', '$start[0]']
          P:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ '$start[2]', '$start[1]', '$start[0]']
)IO_CONFIG";
