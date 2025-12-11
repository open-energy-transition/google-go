Linopy MILP model
=================

Variables:
----------
 * Generator-p (snapshot, Generator)
 * Generator-status (snapshot, Generator-com)
 * Generator-start_up (snapshot, Generator-com)
 * Generator-shut_down (snapshot, Generator-com)
 * Link-p (snapshot, Link)
 * StorageUnit-p_dispatch (snapshot, StorageUnit)
 * StorageUnit-p_store (snapshot, StorageUnit)
 * StorageUnit-state_of_charge (snapshot, StorageUnit)
 * StorageUnit-spill (snapshot, StorageUnit)

Constraints:
------------
 * Generator-fix-p-lower (snapshot, Generator-fix)
 * Generator-fix-p-upper (snapshot, Generator-fix)
 * Generator-com-p-lower (snapshot, Generator-com)
 * Generator-com-p-upper (snapshot, Generator-com)
 * Generator-com-transition-start-up (snapshot, Generator-com)
 * Generator-com-transition-shut-down (snapshot, Generator-com)
 * Generator-com-status-min_up_time_must_stay_up (snapshot, Generator-com)
 * Generator-fix-p-ramp_limit_up (snapshot, Generator-fix)
 * Generator-fix-p-ramp_limit_down (snapshot, Generator-fix)
 * Link-fix-p-lower (snapshot, Link-fix)
 * Link-fix-p-upper (snapshot, Link-fix)
 * StorageUnit-fix-p_dispatch-lower (snapshot, StorageUnit-fix)
 * StorageUnit-fix-p_dispatch-upper (snapshot, StorageUnit-fix)
 * StorageUnit-fix-p_store-lower (snapshot, StorageUnit-fix)
 * StorageUnit-fix-p_store-upper (snapshot, StorageUnit-fix)
 * StorageUnit-fix-state_of_charge-lower (snapshot, StorageUnit-fix)
 * StorageUnit-fix-state_of_charge-upper (snapshot, StorageUnit-fix)
 * StorageUnit-state_of_charge_set (snapshot, StorageUnit-state_of_charge_set_i)
 * Bus-nodal_balance (Bus, snapshot)
 * StorageUnit-energy_balance (snapshot, StorageUnit)
 * StorageUnit_soc_min_constraint (snapshot, StorageUnit)
 * DSR_peak_shaving_max_hours (Generator-com, date)
 * Generator_must_run_capacity (snapshot, Generator)
 * StorageUnit-r_up-limit (snapshot, StorageUnit)
 * StorageUnit-r_down-limit (snapshot, StorageUnit)

Status:
-------
ok