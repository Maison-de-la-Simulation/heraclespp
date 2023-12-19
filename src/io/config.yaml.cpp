#include "config.yaml.hpp"

char const* const io_config = R"IO_CONFIG(
pdi:
  metadata:
    ndim: int
    ndim_u: int
    nfx: int
    max_iter: int
    frequency: int
    iter: int
    gamma: double
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
    grid_communicator: MPI_Comm


  data: # this describes the data that is local to each process
    u:   { type: array, subtype: double, size: ['$ndim_u', '$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]'] }
    rho: { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    P:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    E:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    x:   { type: array, subtype: double, size: ['$nx_glob_ng[0]+2*$n_ghost[0]+1'] }
    y:   { type: array, subtype: double, size: ['$nx_glob_ng[1]+2*$n_ghost[1]+1'] }
    z:   { type: array, subtype: double, size: ['$nx_glob_ng[2]+2*$n_ghost[2]+1'] }
    fx:  { type: array, subtype: double, size: ['$nfx', '$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
    T:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }




  plugins:
    mpi:
    decl_hdf5:
      - file: test_${iter:08}.h5
        collision_policy: replace
        communicator: '$grid_communicator'
        on_event: write_file
        datasets: # this describes the global data (data in the final h5 file)
          ux:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          uy:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          uz:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          rho: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          P:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          E:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          fx:  {type: array, subtype: double, size: ['$nfx', '$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          T:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        write:
          iter:
          current_time:
          gamma:
          u:
            dataset: ux
            memory_selection:
              size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$start[2]', '$start[1]', '$start[0]']
          u:
            when: '$ndim_u>1'
            dataset: uy
            memory_selection:
              size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [1, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$start[2]', '$start[1]', '$start[0]']
          u:
            when: '$ndim_u>2'
            dataset: uz
            memory_selection:
              size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [2, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$start[2]', '$start[1]', '$start[0]']
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
          y:
          z:
          fx:
            when: '$nfx>0'
            memory_selection:
              size: ['$nfx', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nfx', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ 0, '$start[2]', '$start[1]', '$start[0]']
          T:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ '$start[2]', '$start[1]', '$start[0]']

      - file: ${restart_filename}
        on_event: read_file
        communicator: '${grid_communicator}'
        datasets:
          ux:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          uy:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
          uz:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
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
            dataset: ux
            memory_selection:
              size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$start[2]', '$start[1]', '$start[0]']
          u:
            when: '$ndim_u>1'
            dataset: uy
            memory_selection:
              size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [1, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$start[2]', '$start[1]', '$start[0]']
          u:
            when: '$ndim_u>2'
            dataset: uz
            memory_selection:
              size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [2, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$start[2]', '$start[1]', '$start[0]']
          P:
            memory_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ '$start[2]', '$start[1]', '$start[0]']
          fx:
            when: '$nfx>0'
            memory_selection:
              size: ['$nfx', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
            dataset_selection:
              size: ['$nfx', '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
              start: [ 0, '$start[2]', '$start[1]', '$start[0]']
          x:
          y:
          z:
)IO_CONFIG";
