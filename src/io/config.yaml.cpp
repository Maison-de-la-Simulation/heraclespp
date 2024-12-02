#include "config.yaml.hpp"

char const* const io_config = R"PDI_CONFIG(
metadata:
  ndim: int
  nfx: int
  output_id: int
  iter_output_id: int
  time_output_id: int
  iter: int
  gamma: double
  current_time: double
  nx_glob_ng: {type: array, subtype: int, size: 3}
  nx_local_ng: {type: array, subtype: int, size: 3}
  nx_local_wg: {type: array, subtype: int, size: 3}
  n_ghost: {type: array, subtype: int, size: 3}
  start: {type: array, subtype: int, size: 3}
  restart_filename_size: int
  restart_filename: { type: array, subtype: char, size: $restart_filename_size }
  directory_size: int
  directory: { type: array, subtype: char, size: $directory_size }
  output_filename_size: int
  output_filename: { type: array, subtype: char, size: $output_filename_size }
  init_filename_size: int
  init_filename: { type: array, subtype: char, size: $init_filename_size }
  grid_communicator: MPI_Comm
  mpi_rank: int
  ifx: int

data: # this describes the data that is local to each process
  u:   { type: array, subtype: double, size: ['$ndim', '$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]'] }
  rho: { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
  P:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
  E:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
  x:   { type: array, subtype: double, size: ['$nx_glob_ng[0]+2*$n_ghost[0]+1'] }
  y:   { type: array, subtype: double, size: ['$nx_glob_ng[1]+2*$n_ghost[1]+1'] }
  z:   { type: array, subtype: double, size: ['$nx_glob_ng[2]+2*$n_ghost[2]+1'] }
  fx:  { type: array, subtype: double, size: ['$nfx', '$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
  T:   { type: array, subtype: double, size: ['$nx_local_wg[2]', '$nx_local_wg[1]', '$nx_local_wg[0]' ] }
  u_1d:   { type: array, subtype: double, size: ['$nx_local_ng[0]'] }
  rho_1d: { type: array, subtype: double, size: ['$nx_local_ng[0]' ] }
  P_1d:   { type: array, subtype: double, size: ['$nx_local_ng[0]' ] }
  fx_1d:  { type: array, subtype: double, size: ['$nfx', '$nx_local_ng[0]' ] }

plugins:
  mpi:
  decl_hdf5:
    - file: ${directory}//${output_filename}
      on_event: write_replicated_data
      collision_policy: replace
      when: '${mpi_rank}=0'
      datasets:
        x_ng: {type: array, subtype: double, size: '$nx_glob_ng[0]+1' }
        y_ng: {type: array, subtype: double, size: '$nx_glob_ng[1]+1' }
        z_ng: {type: array, subtype: double, size: '$nx_glob_ng[2]+1' }
      write:
        output_id:
        iter_output_id:
        time_output_id:
        iter:
        current_time:
        gamma:
        x:
          dataset: x_ng
          memory_selection:
            size: '$nx_glob_ng[0] + 1'
            start: '$n_ghost[0]'
        y:
          dataset: y_ng
          memory_selection:
            size: '$nx_glob_ng[1] + 1'
            start: '$n_ghost[1]'
        z:
          dataset: z_ng
          memory_selection:
            size: '$nx_glob_ng[2] + 1'
            start: '$n_ghost[2]'
        x:
        y:
        z:
    - file: ${directory}//${output_filename}
      on_event: write_distributed_data
      collision_policy: write_into
      communicator: '$grid_communicator'
      datasets: # this describes the global data (data in the final h5 file)
        ux:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        uy:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        uz:  {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        rho: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        P:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        E:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        T:   {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
      write:
        u:
          dataset: ux
          memory_selection:
            size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: [0, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
          dataset_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['$start[2]', '$start[1]', '$start[0]']
        u:
          when: '$ndim>1'
          dataset: uy
          memory_selection:
            size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: [1, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
          dataset_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['$start[2]', '$start[1]', '$start[0]']
        u:
          when: '$ndim>2'
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
        T:
          memory_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
          dataset_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: [ '$start[2]', '$start[1]', '$start[0]']
    - file: ${directory}//${output_filename}
      on_event: write_fx
      collision_policy: write_into
      communicator: '$grid_communicator'
      datasets: # this describes the global data (data in the final h5 file)
        fx0: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx1: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx2: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx3: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx4: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx5: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx6: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx7: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx8: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
        fx9: {type: array, subtype: double, size: ['$nx_glob_ng[2]', '$nx_glob_ng[1]', '$nx_glob_ng[0]'] }
      write:
        fx:
          dataset: 'fx${ifx}'
          memory_selection:
            size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['${ifx}', '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
          dataset_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['$start[2]', '$start[1]', '$start[0]']

    - file: ${restart_filename}
      on_event: read_file
      communicator: '${grid_communicator}'
      read:
        output_id:
        iter_output_id:
        time_output_id:
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
          when: '$ndim>1'
          dataset: uy
          memory_selection:
            size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: [1, '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
          dataset_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['$start[2]', '$start[1]', '$start[0]']
        u:
          when: '$ndim>2'
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
        x:
        y:
        z:
    - file: ${restart_filename}
      on_event: read_fx
      communicator: '${grid_communicator}'
      read:
        fx:
          dataset: 'fx${ifx}'
          memory_selection:
            size: [1, '$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: ['${ifx}', '$n_ghost[2]', '$n_ghost[1]', '$n_ghost[0]']
          dataset_selection:
            size: ['$nx_local_ng[2]', '$nx_local_ng[1]', '$nx_local_ng[0]']
            start: [ '$start[2]', '$start[1]', '$start[0]']

    - file: ${init_filename}
      on_event: read_hydro_1d
      communicator: '${grid_communicator}'
      read:
        rho_1d:
          dataset_selection:
            size: ['$nx_local_ng[0]']
            start: ['$start[0]']
        u_1d:
          dataset_selection:
            size: ['$nx_local_ng[0]']
            start: ['$start[0]']
        P_1d:
          dataset_selection:
            size: ['$nx_local_ng[0]']
            start: ['$start[0]']
        fx_1d:
          when: '$nfx>0'
          dataset_selection:
            size: ['$nfx', '$nx_local_ng[0]']
            start: [ 0, '$start[0]']
    - file: ${init_filename}
      on_event: read_mesh_1d
      communicator: '${grid_communicator}'
      read:
        x:
)PDI_CONFIG";
