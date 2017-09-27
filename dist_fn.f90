module dist_fn

  use redistribute, only: redist_type

  implicit none

  public :: init_get_fields, finish_get_fields
  public :: get_fields
  public :: init_gxyz
  public :: init_dist_fn, finish_dist_fn
  public :: advance_stella
  public :: time_gke

  private

  interface get_dgdy
     module procedure get_dgdy_4d
     module procedure get_dgdy_2d
  end interface

  interface get_dgdx
     module procedure get_dgdx_4d
     module procedure get_dgdx_2d
  end interface

  interface get_dchidy
     module procedure get_dchidy_4d
     module procedure get_dchidy_2d
  end interface

  interface fill_zed_ghost_zones
     module procedure fill_zed_ghost_zones_real
     module procedure fill_zed_ghost_zones_complex
  end interface

  logical :: get_fields_initialized = .false.
  logical :: get_fields_wstar_initialized = .false.
  logical :: dist_fn_initialized = .false.
  logical :: gxyz_initialized = .false.
  logical :: kp2init = .false.
  logical :: wdriftinit = .false.
  logical :: wstarinit = .false.
  logical :: bessinit = .false.
  logical :: mirrorinit = .false.
  logical :: streaminit = .false.
  logical :: redistinit = .false.
  logical :: readinit = .false.

  integer :: boundary_option_switch
  integer, parameter :: boundary_option_zero = 1, &
       boundary_option_self_periodic = 2, &
       boundary_option_linked = 3

  integer :: adiabatic_option_switch
  integer, parameter :: adiabatic_option_default = 1, &
       adiabatic_option_zero = 2, &
       adiabatic_option_fieldlineavg = 3, &
       adiabatic_option_yavg = 4

  integer :: niter_stream
  real :: stream_errtol
  logical :: fully_explicit, fully_implicit
  logical :: explicit_rk4
  logical :: mirror_explicit, mirror_implicit
  logical :: stream_explicit, stream_implicit
  logical :: wdrifty_explicit, wdrifty_implicit
  logical :: wstar_explicit, wstar_implicit
  real :: xdriftknob, ydriftknob, streamknob, mirrorknob, wstarknob
  real :: stream_upwind
  real :: gamtot_h, gamtot3_h
  real, dimension (:,:,:), allocatable :: gamtot, apar_denom
  real, dimension (:,:,:), allocatable :: gam_stream
  real, dimension (:,:), allocatable :: gamtot3
  complex, dimension (:,:,:), allocatable :: gamtot_wstar, apar_denom_wstar
  complex, dimension (:,:), allocatable :: gamtot3_wstar

!  real, dimension (:,:,:), allocatable :: kperp2

  ! needed for mirror term
  integer, dimension (:,:), allocatable :: mirror_sign
  real, dimension (:,:,:,:), allocatable :: mirror
  real, dimension (:,:,:), allocatable :: mirror_tri_a, mirror_tri_b, mirror_tri_c

  ! needed for parallel streaming term
  integer, dimension (:), allocatable :: stream_sign
  real, dimension (:,:,:), allocatable :: stream
  real, dimension (:,:), allocatable :: stream_tri_a, stream_tri_b, stream_tri_c
  real, dimension (:), allocatable :: stream_tri_diff_a, stream_tri_diff_b, stream_tri_diff_c

  ! these arrays needed to keep track of connections between different
  ! 2pi segments
  integer, dimension (:), allocatable :: neigen
  integer, dimension (:), allocatable :: ig_low, ig_mid, ig_up
  integer, dimension (:,:), allocatable :: nsegments
  integer, dimension (:,:,:), allocatable :: ikxmod
  logical, dimension (:), allocatable :: periodic

  ! geometrical factor multiplying ExB nonlinearity
  real :: nonlin_fac

  ! needed for timing various pieces of gke solve
  real, dimension (2,7) :: time_gke

  type (redist_type) :: kxkyz2vmu
  type (redist_type) :: kxyz2vmu

  logical :: debug = .false.

contains

  subroutine init_get_fields

    use mp, only: sum_allreduce
    use stella_layouts, only: kxkyz_lo
    use stella_layouts, onlY: iz_idx, ikx_idx, iky_idx, is_idx
    use stella_time, only: code_dt
    use dist_fn_arrays, only: aj0v, kperp2
    use run_parameters, only: fphi, fapar
    use run_parameters, only: tite, nine, beta
    use species, only: spec, has_electron_species
    use geometry, only: dl_over_b, gradpar
    use zgrid, only: nzgrid
    use vpamu_grids, only: nvpa, nvgrid, nmu
    use vpamu_grids, only: vpa
    use vpamu_grids, only: maxwellian, integrate_vmu
    use species, only: spec
    use kt_grids, only: naky, nakx, aky, akx

    implicit none

    integer :: ikxkyz, iz, ikx, iky, is
    real :: tmp, wgt
    real, dimension (:,:), allocatable :: g0

    if (get_fields_initialized) return
    get_fields_initialized = .true.

    if (.not.allocated(gamtot)) allocate (gamtot(naky,nakx,-nzgrid:nzgrid)) ; gamtot = 0.
    if (.not.allocated(gamtot3)) then
       if (.not.has_electron_species(spec) &
            .and. adiabatic_option_switch==adiabatic_option_fieldlineavg) then
          allocate (gamtot3(nakx,-nzgrid:nzgrid)) ; gamtot3 = 0.
       else
          allocate (gamtot3(1,1)) ; gamtot3 = 0.
       end if
    end if
    if (.not.allocated(apar_denom)) allocate (apar_denom(naky,nakx,-nzgrid:nzgrid)) ; apar_denom = 0.
    if (.not.allocated(gam_stream)) then
       if (stream_implicit) then
          allocate (gam_stream(naky,naky,-nzgrid:nzgrid))
       else
          allocate (gam_stream(1,1,1))
       end if
       gam_stream = 0.
    end if
       
    if (fphi > epsilon(0.0)) then
       allocate (g0(-nvgrid:nvgrid,nmu))
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          iky = iky_idx(kxkyz_lo,ikxkyz)
          ikx = ikx_idx(kxkyz_lo,ikxkyz)
          iz = iz_idx(kxkyz_lo,ikxkyz)
          is = is_idx(kxkyz_lo,ikxkyz)
          g0 = spread((1.0 - aj0v(:,ikxkyz)**2),1,nvpa)*spread(maxwellian,2,nmu)
          wgt = spec(is)%z*spec(is)%z*spec(is)%dens/spec(is)%temp
          call integrate_vmu (g0, iz, tmp)
          gamtot(iky,ikx,iz) = gamtot(iky,ikx,iz) + tmp*wgt
       end do
       call sum_allreduce (gamtot)
       ! avoid divide by zero when kx=ky=0
       ! do not evolve this mode, so value is irrelevant
       if (aky(1) < epsilon(0.) .and. akx(1) < epsilon(0.)) gamtot(1,1,:) = 1.0

       gamtot_h = sum(spec%z*spec%z*spec%dens/spec%temp)

       if (.not.has_electron_species(spec)) then
          gamtot = gamtot + tite/nine
          gamtot_h = gamtot_h + tite/nine
          if (adiabatic_option_switch == adiabatic_option_fieldlineavg) then
             if (abs(aky(1)) < epsilon(0.)) then
                gamtot3_h = tite/(nine*sum(spec%zt*spec%z*spec%dens))
                do ikx = 1, nakx
                   ! avoid divide by zero for kx=ky=0 mode,
                   ! which we do not need anyway
                   if (abs(akx(ikx)) < epsilon(0.)) cycle
                   tmp = nine/tite-sum(dl_over_b/gamtot(1,ikx,:))
                   gamtot3(ikx,:) = 1./(gamtot(1,ikx,:)*tmp)
                end do
             end if
          end if
       end if

       if (stream_implicit) then
          do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
             iky = iky_idx(kxkyz_lo,ikxkyz)
             ikx = ikx_idx(kxkyz_lo,ikxkyz)
             iz = iz_idx(kxkyz_lo,ikxkyz)
             is = is_idx(kxkyz_lo,ikxkyz)
             g0 = spread(abs(vpa*maxwellian),2,nmu)*spread(aj0v(:,ikxkyz)**2,1,nvpa)
             wgt = spec(is)%z*spec(is)%z*spec(is)%dens*spec(is)%stm/spec(is)%temp &
                  * 0.5*code_dt*gradpar(iz)/gamtot(iky,ikx,iz)
             call integrate_vmu (g0, iz, tmp)
             gam_stream(iky,ikx,iz) = gam_stream(iky,ikx,iz) + tmp*wgt
          end do
       end if

       deallocate (g0)

    end if

    if (fapar > epsilon(0.)) then
       allocate (g0(-nvgrid:nvgrid,nmu))
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          iky = iky_idx(kxkyz_lo,ikxkyz)
          ikx = ikx_idx(kxkyz_lo,ikxkyz)
          iz = iz_idx(kxkyz_lo,ikxkyz)
          is = is_idx(kxkyz_lo,ikxkyz)
          g0 = spread(maxwellian*vpa**2,2,nmu)*spread(aj0v(:,ikxkyz)**2,1,nvpa)
          wgt = 2.0*beta*spec(is)%z*spec(is)%z*spec(is)%dens/spec(is)%mass
          call integrate_vmu (g0, iz, tmp)
          apar_denom(iky,ikx,iz) = apar_denom(iky,ikx,iz) + tmp*wgt
       end do
       call sum_allreduce (apar_denom)
       apar_denom = apar_denom + kperp2

       deallocate (g0)
    end if

    if (wstar_implicit) call init_get_fields_wstar

  end subroutine init_get_fields

  subroutine init_get_fields_wstar

    use constants, only: zi
    use mp, only: sum_allreduce, mp_abort
    use stella_layouts, only: kxkyz_lo
    use stella_layouts, onlY: iz_idx, ikx_idx, iky_idx, is_idx
    use dist_fn_arrays, only: aj0v
    use run_parameters, only: fphi, fapar
    use run_parameters, only: tite, nine, beta
    use stella_time, only: code_dt
    use species, only: spec, has_electron_species
    use geometry, only: dl_over_b
    use zgrid, only: nzgrid
    use vpamu_grids, only: nvpa, nvgrid, nmu
    use vpamu_grids, only: energy
    use vpamu_grids, only: maxwellian, integrate_vmu
    use species, only: spec
    use kt_grids, only: naky, nakx, aky, akx

    implicit none

    integer :: ikxkyz, ig, ikx, iky, is
    complex :: tmp
    complex, dimension (:,:), allocatable :: g0

    if (get_fields_wstar_initialized) return
    get_fields_wstar_initialized = .true.

    if (.not.allocated(gamtot_wstar)) &
         allocate (gamtot_wstar(naky,nakx,-nzgrid:nzgrid))
    gamtot_wstar = 0.
    if (.not.allocated(gamtot3_wstar)) &
         allocate (gamtot3_wstar(nakx,-nzgrid:nzgrid))
    gamtot3_wstar = 0.
    if (.not.allocated(apar_denom_wstar)) &
         allocate (apar_denom_wstar(naky,nakx,-nzgrid:nzgrid))
    apar_denom_wstar = 0.

    if (fphi > epsilon(0.0)) then
       allocate (g0(-nvgrid:nvgrid,nmu))
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          iky = iky_idx(kxkyz_lo,ikxkyz)
          ikx = ikx_idx(kxkyz_lo,ikxkyz)
          ig = iz_idx(kxkyz_lo,ikxkyz)
          is = is_idx(kxkyz_lo,ikxkyz)
          g0 = -zi*aky(iky)*spec(is)%z*spec(is)%dens*spread(aj0v(:,ikxkyz)**2,1,nvpa) &
               ! BACKWARDS DIFFERENCE FLAG
!               *anon(ig,:,:)*0.25*code_dt &
               *spread(maxwellian,2,nmu)*0.5*code_dt &
               *(spec(is)%fprim+spec(is)%tprim*(energy(ig,:,:)-1.5))
          call integrate_vmu (g0, ig, tmp)
          gamtot_wstar(iky,ikx,ig) = gamtot_wstar(iky,ikx,ig) + tmp
       end do
       call sum_allreduce (gamtot_wstar)

       gamtot_wstar = gamtot_wstar + gamtot

       deallocate (g0)

       if (.not.has_electron_species(spec)) then
          ! no need to do anything extra for ky /= 0 because
          ! already accounted for in gamtot_h
          if (adiabatic_option_switch == adiabatic_option_fieldlineavg) then
             if (abs(aky(1)) < epsilon(0.)) then
                do ikx = 1, nakx
                   ! do not need kx=ky=0 mode
                   if (abs(akx(ikx)) < epsilon(0.)) cycle
                   tmp = nine/tite-sum(dl_over_b/gamtot_wstar(1,ikx,:))
                   gamtot3_wstar(ikx,:) = 1./(gamtot_wstar(1,ikx,:)*tmp)
                end do
             end if
          end if
       end if
    end if

    ! FLAG -- NEED TO SORT OUT FINITE FAPAR FOR GSTAR
     if (fapar > epsilon(0.)) then
        write (*,*) 'APAR NOT SETUP FOR GSTAR YET. aborting'
        call mp_abort ('APAR NOT SETUP FOR GSTAR YET. aborting')
     end if
        
!        allocate (g0(-nvgrid:nvgrid,nmu))
!        do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
!           iky = iky_idx(kxkyz_lo,ikxkyz)
!           ikx = ikx_idx(kxkyz_lo,ikxkyz)
!           ig = iz_idx(kxkyz_lo,ikxkyz)
!           is = is_idx(kxkyz_lo,ikxkyz)
!           g0 = spread(vpa**2,2,nmu)*spread(aj0v(:,ikxkyz)**2,1,nvpa)*anon(ig,:,:)
!           wgt = 2.0*beta*spec(is)%z*spec(is)%z*spec(is)%dens/spec(is)%mass
!           call integrate_vmu (g0, ig, tmp)
!           apar_denom(iky,ikx,ig) = apar_denom(iky,ikx,ig) + tmp*wgt
!        end do
!        call sum_allreduce (apar_denom)
!        apar_denom = apar_denom + kperp2

!        deallocate (g0)
!     end if
    
  end subroutine init_get_fields_wstar

  subroutine get_fields (g, phi, apar, dist)

    use mp, only: proc0
    use mp, only: sum_allreduce, mp_abort
    use stella_layouts, only: kxkyz_lo
    use stella_layouts, only: iz_idx, ikx_idx, iky_idx, is_idx
    use dist_fn_arrays, only: aj0v, kperp2
    use run_parameters, only: fphi, fapar
    use run_parameters, only: beta
    use geometry, only: dl_over_b
    use zgrid, only: nzgrid
    use vpamu_grids, only: nvgrid, nvpa, nmu
    use vpamu_grids, only: vpa
    use vpamu_grids, only: integrate_vmu
    use kt_grids, only: nakx, aky
    use species, only: spec, has_electron_species

    implicit none
    
    complex, dimension (-nvgrid:,:,kxkyz_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:), intent (out) :: phi, apar
    character (*), intent (in) :: dist

    real :: wgt
    complex, dimension (:,:), allocatable :: g0
    integer :: ikxkyz, ig, ikx, iky, is
    complex :: tmp

    phi = 0.
    if (fphi > epsilon(0.0)) then
       allocate (g0(-nvgrid:nvgrid,nmu))
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          ig = iz_idx(kxkyz_lo,ikxkyz)
          ikx = ikx_idx(kxkyz_lo,ikxkyz)
          iky = iky_idx(kxkyz_lo,ikxkyz)
          is = is_idx(kxkyz_lo,ikxkyz)
          g0 = spread(aj0v(:,ikxkyz),1,nvpa)*g(:,:,ikxkyz)
          wgt = spec(is)%z*spec(is)%dens
          call integrate_vmu (g0, ig, tmp)
          phi(iky,ikx,ig) = phi(iky,ikx,ig) + wgt*tmp
       end do
       call sum_allreduce (phi)
       if (dist == 'h') then
          phi = phi/gamtot_h
       else if (dist == 'gbar') then
          phi = phi/gamtot
       else if (dist == 'gstar') then
          phi = phi/gamtot_wstar
       else
          if (proc0) write (*,*) 'unknown dist option in get_fields. aborting'
          call mp_abort ('unknown dist option in get_fields. aborting')
       end if

       if (.not.has_electron_species(spec) .and. &
            adiabatic_option_switch == adiabatic_option_fieldlineavg) then
          if (abs(aky(1)) < epsilon(0.)) then
             if (dist == 'h') then
                do ikx = 1, nakx
                   tmp = sum(dl_over_b*phi(1,ikx,:))
                   phi(1,ikx,:) = phi(1,ikx,:) + tmp*gamtot3_h
                end do
             else if (dist == 'gbar') then
                do ikx = 1, nakx
                   tmp = sum(dl_over_b*phi(1,ikx,:))
                   phi(1,ikx,:) = phi(1,ikx,:) + tmp*gamtot3(ikx,:)
                end do
             else if (dist == 'gstar') then
                do ikx = 1, nakx
                   tmp = sum(dl_over_b*phi(1,ikx,:))
                   phi(1,ikx,:) = phi(1,ikx,:) + tmp*gamtot3_wstar(ikx,:)
                end do
             else
                if (proc0) write (*,*) 'unknown dist option in get_fields. aborting'
                call mp_abort ('unknown dist option in get_fields. aborting')
             end if
          end if
       end if

       deallocate (g0)
    end if

    apar = 0.
    if (fapar > epsilon(0.0)) then
       allocate (g0(-nvgrid:nvgrid,nmu))
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          ig = iz_idx(kxkyz_lo,ikxkyz)
          ikx = ikx_idx(kxkyz_lo,ikxkyz)
          iky = iky_idx(kxkyz_lo,ikxkyz)
          is = is_idx(kxkyz_lo,ikxkyz)
          g0 = spread(aj0v(:,ikxkyz),1,nvpa)*spread(vpa,2,nmu)*g(:,:,ikxkyz)
          wgt = 2.0*beta*spec(is)%z*spec(is)%dens*spec(is)%stm
          call integrate_vmu (g0, ig, tmp)
          apar(iky,ikx,ig) = apar(iky,ikx,ig) + tmp*wgt
       end do
       call sum_allreduce (apar)
       if (dist == 'h') then
          apar = apar/kperp2
       else if (dist == 'gbar') then
          apar = apar/apar_denom
       else if (dist == 'gstar') then
          write (*,*) 'APAR NOT SETUP FOR GSTAR YET. aborting.'
          call mp_abort('APAR NOT SETUP FOR GSTAR YET. aborting.')
       else
          if (proc0) write (*,*) 'unknown dist option in get_fields. aborting'
          call mp_abort ('unknown dist option in get_fields. aborting')
       end if
       deallocate (g0)
    end if
    
  end subroutine get_fields

  subroutine finish_get_fields

    implicit none

    get_fields_initialized = .false.
    get_fields_wstar_initialized = .false.
    if (allocated(gamtot)) deallocate (gamtot)
    if (allocated(gamtot3)) deallocate (gamtot3)
    if (allocated(apar_denom)) deallocate (apar_denom)
    if (allocated(gamtot_wstar)) deallocate (gamtot_wstar)
    if (allocated(gamtot3_wstar)) deallocate (gamtot3_wstar)
    if (allocated(apar_denom_wstar)) deallocate (apar_denom_wstar)
    if (allocated(gam_stream)) deallocate (gam_stream)

  end subroutine finish_get_fields

  subroutine init_gxyz

    use dist_fn_arrays, only: gvmu, gold, gnew
    use redistribute, only: gather

    implicit none

    if (gxyz_initialized) return
    gxyz_initialized = .false.

    ! get version of g that has ky,kx,z local
    call gather (kxkyz2vmu, gvmu, gnew)
    gold = gnew

  end subroutine init_gxyz

  subroutine init_dist_fn

    use mp, only: proc0
    use stella_layouts, only: init_stella_layouts, init_dist_fn_layouts
    use stella_transforms, only: init_transforms
    use species, only: init_species
    use species, only: nspec
    use zgrid, only: init_zgrid
    use zgrid, only: nzgrid
    use kt_grids, only: init_kt_grids
    use kt_grids, only: naky, nakx, ny, nx
    use kt_grids, only: alpha_global
    use vpamu_grids, only: init_vpamu_grids
    use vpamu_grids, only: nvgrid, nmu
    use run_parameters, only: init_run_parameters
    use run_parameters, only: nonlinear
    use neoclassical_terms, only: init_neoclassical_terms

    implicit none

    if (dist_fn_initialized) return
    dist_fn_initialized = .true.

    debug = debug .and. proc0
    
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_stella_layouts'
    call init_stella_layouts
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_species'
    call init_species
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_zgrid'
    call init_zgrid
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_kt_grids'
    call init_kt_grids
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_vpamu_grids'
    call init_vpamu_grids
    if (debug) write (*,*) 'dist_fn::init_dist_fn::read_parameters'
    call read_parameters
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_run_parameters'
    call init_run_parameters
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_dist_fn_layouts'
    call init_dist_fn_layouts (nzgrid, naky, nakx, nvgrid, nmu, nspec, ny, nx)
    if (debug) write (*,*) 'dist_fn::init_dist_fn::allocate_arrays'
    call allocate_arrays
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_kperp2'
    call init_kperp2
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_connections'
    call init_connections
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_vperp2'
    call init_vperp2
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_bessel'
    call init_bessel
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_mirror'
    call init_mirror
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_parstream'
    call init_parstream
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_wdrift'
    call init_wdrift
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_wstar'
    call init_wstar
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_ExB_nonlinearity'
    if (nonlinear) call init_ExB_nonlinearity
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_neoclassical_terms'
    call init_neoclassical_terms
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_redistribute'
    call init_redistribute
    if (debug) write (*,*) 'dist_fn::init_dist_fn::init_cfl'
    call init_cfl
    if (nonlinear .or. alpha_global) then
       if (debug) write (*,*) 'dist_fn::init_dist_fn::init_transforms'
       call init_transforms
    end if

  end subroutine init_dist_fn

  subroutine read_parameters

    use file_utils, only: input_unit, error_unit, input_unit_exist
    use geometry, only: geo_surf
    use zgrid, only: shat_zero
    use text_options, only: text_option, get_option_value
    use species, only: nspec
    use mp, only: proc0, broadcast

    implicit none

    logical :: dfexist

    type (text_option), dimension (6), parameter :: boundaryopts = &
         (/ text_option('default', boundary_option_zero), &
            text_option('zero', boundary_option_zero), &
            text_option('unconnected', boundary_option_zero), &
            text_option('self-periodic', boundary_option_self_periodic), &
            text_option('periodic', boundary_option_self_periodic), &
            text_option('linked', boundary_option_linked) /)
    character(20) :: boundary_option

    type (text_option), dimension (7), parameter :: adiabaticopts = &
         (/ text_option('default', adiabatic_option_default), &
            text_option('no-field-line-average-term', adiabatic_option_default), &
            text_option('field-line-average-term', adiabatic_option_fieldlineavg), &
            text_option('iphi00=0', adiabatic_option_default), &
            text_option('iphi00=1', adiabatic_option_default), &
            text_option('iphi00=2', adiabatic_option_fieldlineavg), &
            text_option('iphi00=3', adiabatic_option_yavg)/)
    character(30) :: adiabatic_option
            
    namelist /dist_fn_knobs/ boundary_option, &
         xdriftknob, ydriftknob, streamknob, mirrorknob, wstarknob, &
         mirror_explicit, stream_explicit, wdrifty_explicit, wstar_explicit, &
         adiabatic_option, niter_stream, stream_errtol, explicit_rk4, &
         stream_upwind
    integer :: ierr, in_file

    if (readinit) return
    readinit = .true.

    if (proc0) then
       boundary_option = 'default'
       adiabatic_option = 'default'
       xdriftknob = 1.0
       ydriftknob = 1.0
       streamknob = 1.0
       mirrorknob = 1.0
       wstarknob = 1.0
       stream_upwind = 1.0
       mirror_explicit = .false.
       stream_explicit = .false.
       wdrifty_explicit = .true.
       wstar_explicit = .true.
       explicit_rk4 = .false.
       niter_stream = 2
       stream_errtol = 0.001

       in_file = input_unit_exist("dist_fn_knobs", dfexist)
       if (dfexist) read (unit=in_file, nml=dist_fn_knobs)

       if(abs(geo_surf%shat) <=  shat_zero) boundary_option = 'periodic'

       ierr = error_unit()
       call get_option_value &
            (boundary_option, boundaryopts, boundary_option_switch, &
            ierr, "boundary_option in dist_fn_knobs")

       call get_option_value &
            (adiabatic_option, adiabaticopts, adiabatic_option_switch, &
            ierr, "adiabatic_option in dist_fn_knobs")

    end if

    call broadcast (boundary_option_switch)
    call broadcast (adiabatic_option_switch)
    call broadcast (xdriftknob)
    call broadcast (ydriftknob)
    call broadcast (streamknob)
    call broadcast (mirrorknob)
    call broadcast (wstarknob)
    call broadcast (mirror_explicit)
    call broadcast (stream_explicit)
    call broadcast (wdrifty_explicit)
    call broadcast (wstar_explicit)
    call broadcast (explicit_rk4)
    call broadcast (niter_stream)
    call broadcast (stream_errtol)
    call broadcast (stream_upwind)

    mirror_implicit = .not.mirror_explicit
    stream_implicit = .not.stream_explicit
    wdrifty_implicit = .not.wdrifty_explicit
    wstar_implicit = .not.wstar_explicit

    ! not all terms have implicit options (yet?)
    fully_implicit = .false.

    if (mirror_explicit .and. stream_explicit &
         .and. wdrifty_explicit .and. wstar_explicit) then
       fully_explicit = .true.
    else
       fully_explicit = .false.
    end if

  end subroutine read_parameters 

  subroutine init_kperp2

    use dist_fn_arrays, only: kperp2
    use geometry, only: gds2, gds21, gds22
    use geometry, only: geo_surf
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx, theta0
    use kt_grids, only: akx, aky

    implicit none

    integer :: iky, ikx

    if (kp2init) return
    kp2init = .true.

    allocate (kperp2(naky,nakx,-nzgrid:nzgrid))
    do iky = 1, naky
       if (abs(aky(iky)) < epsilon(0.)) then
          do ikx = 1, nakx
             kperp2(iky,ikx,:) = akx(ikx)*akx(ikx)*gds22/(geo_surf%shat**2)
          end do
       else
          do ikx = 1, nakx
             kperp2(iky,ikx,:) = aky(iky)*aky(iky) &
                  *(gds2 + 2.0*theta0(iky,ikx)*gds21 &
                  + theta0(iky,ikx)*theta0(iky,ikx)*gds22)
          end do
       end if
    end do

  end subroutine init_kperp2

  subroutine init_wdrift

    use dist_fn_arrays, only: wdriftx, wdrifty
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: iv_idx, imu_idx, is_idx
    use stella_time, only: code_dt
    use species, only: spec
    use zgrid, only: nzgrid
    use geometry, only: cvdrift, gbdrift
    use geometry, only: cvdrift0, gbdrift0
    use geometry, only: geo_surf
    use vpamu_grids, only: vpa, vperp2
    use kt_grids, only: ny_ffs

    implicit none

    integer :: ivmu, iv, imu, is, iy

    if (wdriftinit) return
    wdriftinit = .true.

    if (.not.allocated(wdriftx)) &
         allocate (wdriftx(ny_ffs,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    if (.not.allocated(wdrifty)) &
         allocate (wdrifty(ny_ffs,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))

    ! FLAG -- need to deal with shat=0 case.  ideally move away from q as x-coordinate
    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       iv = iv_idx(vmu_lo,ivmu)
       imu = imu_idx(vmu_lo,ivmu)
       is = is_idx(vmu_lo,ivmu)
       do iy = 1, ny_ffs
          wdrifty(iy,:,ivmu) = -ydriftknob*0.5*code_dt*spec(is)%tz &
               * (cvdrift*vpa(iv)**2 + gbdrift*0.5*vperp2(:,imu))
          wdriftx(iy,:,ivmu) = -xdriftknob*0.5*code_dt*spec(is)%tz/geo_surf%shat &
               * (cvdrift0*vpa(iv)**2 + gbdrift0*0.5*vperp2(:,imu))
       end do
    end do

  end subroutine init_wdrift

  subroutine init_wstar

    use dist_fn_arrays, only: wstar
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: iv_idx, imu_idx, is_idx
    use stella_time, only: code_dt
    use species, only: spec
    use zgrid, only: nzgrid
    use vpamu_grids, only: energy, maxwellian

    implicit none

    integer :: is, imu, iv, ivmu

    if (wstarinit) return
    wstarinit = .true.

    if (.not.allocated(wstar)) &
         allocate (wstar(-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc)) ; wstar = 0.0

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       is = is_idx(vmu_lo,ivmu)
       imu = imu_idx(vmu_lo,ivmu)
       iv = iv_idx(vmu_lo,ivmu)
       wstar(:,ivmu) = 0.5*wstarknob*0.5*code_dt*maxwellian(iv) &
            * (spec(is)%fprim+spec(is)%tprim*(energy(:,iv,imu)-1.5))
    end do

  end subroutine init_wstar

  subroutine init_ExB_nonlinearity

    use geometry, only: geo_surf, drhodpsi

    implicit none

    nonlin_fac = 0.5*geo_surf%qinp/(geo_surf%rhoc*drhodpsi)

  end subroutine init_ExB_nonlinearity

  subroutine allocate_arrays

    use stella_layouts, only: kxkyz_lo, vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx
    use vpamu_grids, only: nvgrid, nmu
    use dist_fn_arrays, only: gnew, gold
    use dist_fn_arrays, only: g1, g2, g3
    use dist_fn_arrays, only: gvmu

    implicit none

    if (.not.allocated(gnew)) &
         allocate (gnew(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    gnew = 0.
    if (.not.allocated(gold)) &
         allocate (gold(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    gold = 0.
    if (.not.allocated(g1)) &
         allocate (g1(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    g1 = 0.
    if (.not.allocated(g2)) &
         allocate (g2(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    g2 = 0.
    if (.not.allocated(g3) .and. explicit_rk4) then
       allocate (g3(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
       g3 = 0.
    else
       allocate (g3(1,1,1,1))
    end if
    if (.not.allocated(gvmu)) &
         allocate (gvmu(-nvgrid:nvgrid,nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
    gvmu = 0.

  end subroutine allocate_arrays

  subroutine init_connections

    use zgrid, only: nperiod, nzgrid, nzed
    use kt_grids, only: ntheta0, nakx, naky
    use kt_grids, only: jtwist_out, aky
    use species, only: nspec
    use vpamu_grids, only: nmu, nvgrid

    implicit none

    integer :: iseg, iky, ie, ntg, ikx, ikxshiftend
    integer :: nseg_max, neigen_max
    integer :: iky_max
    integer, dimension (:), allocatable :: ikx_shift_left_kypos, ikx_shift_left_kyneg
    integer, dimension (:,:), allocatable :: ikx_shift

    ntg = nzed/2
    ! iky_max is the index of the most positive ky
    iky_max = naky/2+1

    if (.not. allocated(neigen)) allocate (neigen(naky))
    if (.not. allocated(periodic)) allocate (periodic(naky)) ; periodic = .false.

    if (boundary_option_switch==boundary_option_self_periodic) then
       periodic = .true.
    else
       where (abs(aky) < epsilon(0.0)) periodic = .true.
    end if

    select case (boundary_option_switch)
    case (boundary_option_linked)

       ! ntheta0 = 2*(nakx-1) + 1
       ! nakx includes kx >= 0
       ! ntheta0 also includes kx < 0
       neigen(1) = ntheta0
       if (naky > 1) then
          do iky = 2, iky_max
             ! must link different kx values at theta = +/- pi
             ! neigen is the number of independent eigenfunctions along the field line
             neigen(iky) = min((iky-1)*jtwist_out,ntheta0)
          end do
          ! number of eigenfunctions for -ky is same as for +ky
          neigen(iky_max+1:) = neigen(iky_max:2:-1)
       end if

       neigen_max = maxval(neigen)

       if (.not. allocated(ikx_shift_left_kypos)) then
          allocate (ikx_shift_left_kypos(neigen_max)) ; ikx_shift_left_kypos = 0
          allocate (ikx_shift_left_kyneg(neigen_max)) ; ikx_shift_left_kyneg = 0
          allocate (ikx_shift(ntheta0,naky)) ; ikx_shift = 0
       end if

       ! figure out how much to shift ikx by to get to
       ! the left-most (theta-theta0) in each set of connected 2pi segments
       ! note that theta0 goes from 0 to theta0_max and then from theta0_min back
       ! to -dtheta0
       do ikx = 1, neigen_max
          ! first ntheta0/2+1 theta0s are 0 and all positive theta0 values
          ! remainder are negative theta0s
          ! note that ntheta0 is always positive for box
          ! theta_0 = kx / ky / shat
          ! if ky > 0, then most positive theta_0 corresponds to most positive kx
          if (ikx <= nakx) then
             ikx_shift_left_kypos(ikx) = nakx-2*ikx+1
          else
             ikx_shift_left_kypos(ikx) = 3*nakx-2*ikx
          end if
          ! if ky < 0, most positive theta_0 corresponds to most negative kx
          if (ikx < nakx) then
             ikx_shift_left_kyneg(ikx) = nakx
          else
             ikx_shift_left_kyneg(ikx) = 1-nakx
          end if
       end do

       do iky = 1, naky
          ! ikx_shift is how much to shift each ikx by to connect
          ! to the next theta0 (from most positive to most negative)

          ! if ky < 0, then going to more negative theta0
          ! corresponds to going to more positive kx
          if (aky(iky) < 0.0) then
             ! first treat kx positive
             ! connect to more positive kx neigen away
             ! but only if not trying to connect to kx
             ! so positive that ikx is not on grid
             if (nakx - neigen(iky) > 0) ikx_shift(:nakx-neigen(iky),iky) = neigen(iky)
             ! next treat kx negative
             ! if kx sufficiently negative, then 
             ! shifting by neigen keeps kx negative
             do ikx = nakx+1, ntheta0
                if (ikx+neigen(iky) <= ntheta0) then
                   ikx_shift(ikx,iky) = neigen(iky)
                   ! if theta0 not sufficiently negative,
                   ! then must shift to postive theta0
                else if (ikx+neigen(iky) <= ntheta0+nakx) then
                   ikx_shift(ikx,iky) = neigen(iky) - ntheta0
                end if
             end do
          else
             ! if ky > 0, then going to more negative theta0
             ! corresponds to going to more negative kx
             do ikx = 1, nakx
                ! if theta0 is sufficiently positive, shifting to more
                ! negative theta0 corresponds to decreasing ikx
                if (ikx-neigen(iky) > 0) then
                   ikx_shift(ikx,iky) = -neigen(iky)
                   ! if a positive theta0 connects to a negative theta0
                   ! must do more complicated mapping of ikx
                else if (ikx-neigen(iky)+ntheta0 >= nakx+1) then
                   ikx_shift(ikx,iky) = ntheta0 - neigen(iky)
                end if
             end do
             ! if theta0 is negative, then shifting to more negative
             ! theta0 corresponds to decreasing ikx
             do ikx = nakx+1, ntheta0
                ! if theta0 is sufficiently negative, it has no
                ! more negative theta0 with which it can connect
                if (ikx-neigen(iky) >= nakx) then
                   ikx_shift(ikx,iky) = -neigen(iky)
                end if
                ! theta0 is positive
             end  do
          end if
       end do

       if (.not. allocated(nsegments)) then
          allocate (nsegments(neigen_max,naky))
       end if

       do iky = 1, naky
          if (neigen(iky) == 0) then
             nsegments(:,iky) = 1
          else
             nsegments(:,iky) = (ntheta0-1)/neigen(iky)

             do ie = 1, mod(ntheta0-1,neigen(iky))+1
                nsegments(ie,iky) = nsegments(ie,iky) + 1
             end do
          end if
       end do

       nseg_max = maxval(nsegments)

       if (.not. allocated(ig_low)) then
          allocate (ig_low(nseg_max)) ; ig_low = -nzgrid
          allocate (ig_mid(nseg_max)) ; ig_mid = 0
          allocate (ig_up(nseg_max)) ; ig_up = nzgrid
       end if
       
    case default
       
       neigen = ntheta0 ; neigen_max = ntheta0
       
       if (.not. allocated(ikx_shift_left_kypos)) then
          allocate (ikx_shift_left_kypos(neigen_max))
          allocate (ikx_shift_left_kyneg(neigen_max))
          allocate (ikx_shift(ntheta0,naky))
       end if
       ikx_shift = 0 ; ikx_shift_left_kypos = 0 ; ikx_shift_left_kyneg = 0
       
       if (.not. allocated(nsegments)) then
          allocate (nsegments(neigen_max,naky))
       end if
       
       ! this is the number of 2pi poloidal segments in the extended theta domain,
       ! which is needed in initializing the reponse matrix and doing the implicit sweep
       nsegments = 2*(nperiod-1) + 1
       
       nseg_max = maxval(nsegments)
       
       if (.not. allocated(ig_low)) then
          allocate (ig_low(nseg_max))
          allocate (ig_mid(nseg_max))
          allocate (ig_up(nseg_max))
       end if

       ! ig_low(j) is the ig index corresponding to the inboard midplane from below (theta=-pi) within the jth segment
       ! ig_mid(j) is the ig index corresponding to the outboard midplane (theta=0) within the jth segment
       do iseg = 1, nseg_max
          ig_low(iseg) = -nzgrid + (iseg-1)*nzed
          ig_mid(iseg) = ig_low(iseg) + nzed/2
          ig_up(iseg) = ig_low(iseg) + nzed
       end do

    end select

    if (.not. allocated(ikxmod)) allocate (ikxmod(nseg_max,neigen_max,naky))
    do iky = 1, naky
       ! only do the following once for each independent set of theta0s
       ! the assumption here is that all kx are on processor and sequential
       do ikx = 1, neigen(iky)
          if (aky(iky) < 0.) then
             ikxshiftend = ikx_shift_left_kyneg(ikx)
          else
             ikxshiftend = ikx_shift_left_kypos(ikx)
          end if
          ! remap to start at theta0 = theta0_max
          ! (so that theta-theta0 is most negative)
          ! for this set of connected theta0s
          iseg = 1
          ikxmod(iseg,ikx,iky) = ikx + ikxshiftend
          if (nsegments(ikx,iky) > 1) then
             do iseg = 2, nsegments(ikx,iky)
                ikxmod(iseg,ikx,iky) = ikxmod(iseg-1,ikx,iky) + ikx_shift(ikxmod(iseg-1,ikx,iky),iky)
             end do
          end if
       end do
    end do

    if (allocated(ikx_shift_left_kypos)) deallocate (ikx_shift_left_kypos)
    if (allocated(ikx_shift_left_kyneg)) deallocate (ikx_shift_left_kyneg)
    if (allocated(ikx_shift)) deallocate (ikx_shift)

  end subroutine init_connections

  subroutine init_vperp2

    use geometry, only: bmag
    use zgrid, only: nzgrid
    use vpamu_grids, only: vperp2, energy
    use vpamu_grids, only: vpa, mu
    use vpamu_grids, only: nmu, nvgrid

    implicit none

    integer :: iv
    
    if (.not.allocated(vperp2)) allocate (vperp2(-nzgrid:nzgrid,nmu)) ; vperp2 = 0.
    if (.not.allocated(energy)) allocate (energy(-nzgrid:nzgrid,-nvgrid:nvgrid,nmu)) ; energy = 0.

    vperp2 = 2.0*spread(mu,1,2*nzgrid+1)*spread(bmag,2,nmu)

    do iv = -nvgrid, nvgrid
       energy(:,iv,:) = vpa(iv)**2 + 2.0*spread(mu,1,2*nzgrid+1)*spread(bmag,2,nmu)
    end do

  end subroutine init_vperp2

  subroutine init_bessel

    use dist_fn_arrays, only: aj0v, aj1v
    use dist_fn_arrays, only: aj0x
    use dist_fn_arrays, only: kperp2
    use species, only: spec, nspec
    use geometry, only: bmag
    use zgrid, only: nzgrid
    use vpamu_grids, only: vperp2, nmu
    use kt_grids, only: naky, nakx
    use stella_layouts, only: kxkyz_lo, vmu_lo
    use stella_layouts, only: iky_idx, ikx_idx, iz_idx, is_idx, imu_idx
    use spfunc, only: j0, j1

    implicit none

    integer :: ig, iky, ikx, imu, is
    integer :: ikxkyz, ivmu
    real :: arg

    if (bessinit) return
    bessinit = .true.

    call init_kperp2

    if (.not.allocated(aj0v)) then
       allocate (aj0v(nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
       aj0v = 0.
    end if
    if (.not.allocated(aj0x)) then
       allocate (aj0x(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
       aj0x = 0.
    end if
    if (.not.allocated(aj1v)) then
       allocate (aj1v(nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
       aj1v = 0.
    end if
    
    do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
       iky = iky_idx(kxkyz_lo,ikxkyz)
       ikx = ikx_idx(kxkyz_lo,ikxkyz)
       ig = iz_idx(kxkyz_lo,ikxkyz)
       is = is_idx(kxkyz_lo,ikxkyz)
       do imu = 1, nmu
          arg = spec(is)%smz*sqrt(vperp2(ig,imu)*kperp2(iky,ikx,ig))/bmag(ig)
          aj0v(imu,ikxkyz) = j0(arg)
          ! note that j1 returns and aj1 stores J_1(x)/x (NOT J_1(x)), 
          aj1v(imu,ikxkyz) = j1(arg)
       end do
    end do

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       is = is_idx(vmu_lo,ivmu)
       imu = imu_idx(vmu_lo,ivmu)
       do ig = -nzgrid, nzgrid
          do ikx = 1, nakx
             do iky = 1, naky
                arg = spec(is)%smz*sqrt(vperp2(ig,imu)*kperp2(iky,ikx,ig))/bmag(ig)
                aj0x(iky,ikx,ig,ivmu) = j0(arg)
             ! note that j1 returns and aj1 stores J_1(x)/x (NOT J_1(x)), 
!             aj1(ig,ivmu) = j1(arg)
             end do
          end do
       end do
    end do

  end subroutine init_bessel

  subroutine init_mirror

    use stella_time, only: code_dt
    use species, only: spec, nspec
    use vpamu_grids, only: nmu
    use vpamu_grids, only: mu
    use zgrid, only: nzgrid
    use kt_grids, only: ny_ffs
    use geometry, only: dbdthet, gradpar
!    use sherman_morrison, only: init_invert_mirror_operator

    implicit none

    integer :: iz, iy, imu, is

    if (mirrorinit) return
    mirrorinit = .true.
    
    if (.not.allocated(mirror)) allocate (mirror(ny_ffs,-nzgrid:nzgrid,nmu,nspec)) ; mirror = 0.
    if (.not.allocated(mirror_sign)) allocate (mirror_sign(ny_ffs,-nzgrid:nzgrid)) ; mirror_sign = 0

    do is = 1, nspec
       do imu = 1, nmu
          do iy = 1, ny_ffs
             do iz = -nzgrid, nzgrid
                mirror(iy,iz,imu,is) = mirrorknob*code_dt*spec(is)%stm &
                     *mu(imu)*dbdthet(iz)*gradpar(iz)*0.5
             end do
          end do
       end do
    end do

    do iy = 1, ny_ffs
       ! mirror_sign set to +/- 1 depending on the sign of the mirror term.
       ! NB: mirror_sign = -1 corresponds to positive advection velocity
       do iz = -nzgrid, nzgrid
          mirror_sign(iy,iz) = int(sign(1.0,mirror(iy,iz,1,1)))
       end do
    end do

    if (mirror_implicit) call init_invert_mirror_operator

  end subroutine init_mirror

  subroutine init_invert_mirror_operator

    use mp, only: mp_abort
    use stella_layouts, only: kxkyz_lo
    use stella_layouts, only: iz_idx, is_idx
    use vpamu_grids, only: dvpa
    use vpamu_grids, only: nvgrid, nmu
    use kt_grids, only: alpha_global

    implicit none

    integer :: sgn
    integer :: ikxkyz, iy, iz, is, imu, iv
    real, dimension (:,:), allocatable :: a, b, c

    allocate (a(-nvgrid:nvgrid,-1:1)) ; a = 0.
    allocate (b(-nvgrid:nvgrid,-1:1)) ; b = 0.
    allocate (c(-nvgrid:nvgrid,-1:1)) ; c = 0.

    if (.not.allocated(mirror_tri_a)) then
       allocate(mirror_tri_a(-nvgrid:nvgrid,nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc)) ; mirror_tri_a = 0.
       allocate(mirror_tri_b(-nvgrid:nvgrid,nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc)) ; mirror_tri_b = 0.
       allocate(mirror_tri_c(-nvgrid:nvgrid,nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc)) ; mirror_tri_c = 0.
    end if

    ! corresponds to sign of mirror term positive on RHS of equation
    b(:,1) = -1./dvpa
    c(:nvgrid-1,1) = 1./dvpa
    ! corresponds to sign of mirror term negative on RHS of equation
    a(-nvgrid+1:,-1) = -1./dvpa
    b(:,-1) = 1./dvpa

    if (alpha_global) then
       write (*,*) 'not yet setup for alpha_global'
       call mp_abort ('mirror not yet setup fo alpha_global')
    else
       ! multiply by mirror coefficient
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          iy = 1
          iz = iz_idx(kxkyz_lo,ikxkyz)
          is = is_idx(kxkyz_lo,ikxkyz)
          sgn = mirror_sign(iy,iz)
          do imu = 1, nmu
             do iv = -nvgrid, nvgrid
! BACKWARDS DIFFERENCE FLAG
                mirror_tri_a(iv,imu,ikxkyz) = -2.0*a(iv,sgn)*mirror(iy,iz,imu,is)
                mirror_tri_b(iv,imu,ikxkyz) = 1.0-2.0*b(iv,sgn)*mirror(iy,iz,imu,is)
                mirror_tri_c(iv,imu,ikxkyz) = -2.0*c(iv,sgn)*mirror(iy,iz,imu,is)
             end do
          end do
       end do
    end if

    deallocate (a, b, c)

  end subroutine init_invert_mirror_operator

  subroutine init_parstream

    use finite_differences, only: fd3pt
    use stella_time, only: code_dt
    use species, only: spec, nspec
    use vpamu_grids, only: nvgrid, nvpa
    use vpamu_grids, only: vpa
    use zgrid, only: nzgrid
    use geometry, only: gradpar

    implicit none

    integer :: iv

    if (streaminit) return
    streaminit = .true.

    if (.not.allocated(stream)) allocate (stream(-nzgrid:nzgrid,-nvgrid:nvgrid,nspec)) ; stream = 0.
    if (.not.allocated(stream_sign)) allocate (stream_sign(-nvgrid:nvgrid)) ; stream_sign = 0

    ! sign of stream corresponds to appearing on RHS of GK equation
    stream = -streamknob*code_dt*spread(spread(spec%stm,1,2*nzgrid+1),2,nvpa) &
         * spread(spread(vpa,1,2*nzgrid+1)*spread(gradpar,2,nvpa),3,nspec)*0.5

    ! stream_sign set to +/- 1 depending on the sign of the parallel streaming term.
    ! NB: stream_sign = -1 corresponds to positive advection velocity
    do iv = -nvgrid, nvgrid
       stream_sign(iv) = int(sign(1.0,stream(0,iv,1)))
    end do

    if (stream_implicit) call init_invert_stream_operator

  end subroutine init_parstream

  subroutine init_invert_stream_operator

    use zgrid, only: delzed

    implicit none

    integer :: nz, nseg_max

    nz = maxval(ig_up-ig_low)
    nseg_max = maxval(nsegments)

    if (.not.allocated(stream_tri_a)) then
       allocate (stream_tri_a(nz*nseg_max+1,-1:1)) ; stream_tri_a = 0.
       allocate (stream_tri_b(nz*nseg_max+1,-1:1)) ; stream_tri_b = 0.
       allocate (stream_tri_c(nz*nseg_max+1,-1:1)) ; stream_tri_c = 0.
    end if
    if (.not.allocated(stream_tri_diff_a)) then
       allocate (stream_tri_diff_a(nz*nseg_max+1)) ; stream_tri_diff_a = 0.
       allocate (stream_tri_diff_b(nz*nseg_max+1)) ; stream_tri_diff_b = 0.
       allocate (stream_tri_diff_c(nz*nseg_max+1)) ; stream_tri_diff_c = 0.
    end if

    ! corresponds to sign of stream term positive on RHS of equation
    ! FLAG -- ASSUMING EQUAL SPACING IN Z
    stream_tri_a(2:,1) = -0.5*(1.0-stream_upwind)/delzed(0)
    stream_tri_b(2:,1) = -stream_upwind/delzed(0)
    stream_tri_c(2:,1) = (1.0+stream_upwind)*0.5/delzed(0)
    ! must treat boundary carefully
    stream_tri_b(1,1) = -1.0/delzed(0)
    stream_tri_c(1,1) = 1.0/delzed(0)
    ! corresponds to sign of stream term negative on RHS of equation
    ! FLAG -- ASSUMING EQUAL SPACING IN Z
    stream_tri_a(:nz*nseg_max,-1) = -0.5*(1.0+stream_upwind)/delzed(0)
    stream_tri_b(:nz*nseg_max,-1) = stream_upwind/delzed(0)
    stream_tri_c(:nz*nseg_max,-1) = 0.5*(1.0-stream_upwind)/delzed(0)
    ! must treat boundary carefully
    stream_tri_a(nz*nseg_max+1,-1) = -1.0/delzed(0)
    stream_tri_b(nz*nseg_max+1,-1) = 1.0/delzed(0)

    stream_tri_diff_a = stream_tri_a(:,-1)-stream_tri_a(:,1)
    stream_tri_diff_b = stream_tri_b(:,-1)-stream_tri_b(:,1)
    stream_tri_diff_c = stream_tri_c(:,-1)-stream_tri_c(:,1)

  end subroutine init_invert_stream_operator

  subroutine init_redistribute

    use kt_grids, only: alpha_global

    implicit none

    if (redistinit) return
    redistinit = .true.

    if (debug) write (*,*) 'dist_fn::init_redistribute::init_kxkyz_to_gzkxky_redistribute'
    call init_kxkyz_to_vmu_redistribute
    if (alpha_global) call init_kxyz_to_vmu_redistribute

  end subroutine init_redistribute

  subroutine init_kxkyz_to_vmu_redistribute

    use mp, only: nproc
    use stella_layouts, only: kxkyz_lo, vmu_lo
    use stella_layouts, only: kxkyzidx2vmuidx
    use stella_layouts, only: idx_local, proc_id
    use redistribute, only: index_list_type, init_redist
    use redistribute, only: delete_list, set_redist_character_type
    use vpamu_grids, only: nvgrid, nmu
    use zgrid, only: nzgrid

    implicit none

    type (index_list_type), dimension (0:nproc-1) :: to_list, from_list
    integer, dimension (0:nproc-1) :: nn_to, nn_from
    integer, dimension (3) :: from_low, from_high
    integer, dimension (4) :: to_high, to_low
    integer :: ikxkyz, ivmu
    integer :: iv, imu, iky, ikx, ig
    integer :: ip, n
    logical :: initialized = .false.

    if (initialized) return
    initialized = .true.

    ! count number of elements to be redistributed to/from each processor
    nn_to = 0
    nn_from = 0
    do ikxkyz = kxkyz_lo%llim_world, kxkyz_lo%ulim_world
       do imu = 1, nmu
          do iv = -nvgrid, nvgrid
             call kxkyzidx2vmuidx (iv, imu, ikxkyz, kxkyz_lo, vmu_lo, iky, ikx, ig, ivmu)
             if (idx_local(kxkyz_lo,ikxkyz)) &
                  nn_from(proc_id(vmu_lo,ivmu)) = nn_from(proc_id(vmu_lo,ivmu)) + 1
             if (idx_local(vmu_lo,ivmu)) &
                  nn_to(proc_id(kxkyz_lo,ikxkyz)) = nn_to(proc_id(kxkyz_lo,ikxkyz)) + 1
          end do
       end do
    end do
    
    do ip = 0, nproc-1
       if (nn_from(ip) > 0) then
          allocate (from_list(ip)%first(nn_from(ip)))
          allocate (from_list(ip)%second(nn_from(ip)))
          allocate (from_list(ip)%third(nn_from(ip)))
       end if
       if (nn_to(ip) > 0) then
          allocate (to_list(ip)%first(nn_to(ip)))
          allocate (to_list(ip)%second(nn_to(ip)))
          allocate (to_list(ip)%third(nn_to(ip)))
          allocate (to_list(ip)%fourth(nn_to(ip)))
       end if
    end do

    ! get local indices of elements distributed to/from other processors
    nn_to = 0
    nn_from = 0

    ! loop over all vmu indices, find corresponding y indices
    do ikxkyz = kxkyz_lo%llim_world, kxkyz_lo%ulim_world
       do imu = 1, nmu
          do iv = -nvgrid, nvgrid
             ! obtain corresponding y indices
             call kxkyzidx2vmuidx (iv, imu, ikxkyz, kxkyz_lo, vmu_lo, iky, ikx, ig, ivmu)
             ! if vmu index local, set:
             ! ip = corresponding y processor
             ! from_list%first-third arrays = iv,imu,ikxkyz  (ie vmu indices)
             ! later will send from_list to proc ip
             if (idx_local(kxkyz_lo,ikxkyz)) then
                ip = proc_id(vmu_lo,ivmu)
                n = nn_from(ip) + 1
                nn_from(ip) = n
                from_list(ip)%first(n) = iv
                from_list(ip)%second(n) = imu
                from_list(ip)%third(n) = ikxkyz
             end if
             ! if y index local, set ip to corresponding vmu processor
             ! set to_list%first,second arrays = iky,iy  (ie y indices)
             ! will receive to_list from ip
             if (idx_local(vmu_lo,ivmu)) then
                ip = proc_id(kxkyz_lo,ikxkyz)
                n = nn_to(ip) + 1
                nn_to(ip) = n
                to_list(ip)%first(n) = iky
                to_list(ip)%second(n) = ikx
                to_list(ip)%third(n) = ig
                to_list(ip)%fourth(n) = ivmu
             end if
          end do
       end do
    end do

    from_low(1) = -nvgrid
    from_low(2) = 1
    from_low(3) = kxkyz_lo%llim_proc

    from_high(1) = nvgrid
    from_high(2) = nmu
    from_high(3) = kxkyz_lo%ulim_alloc

    to_low(1) = 1
    to_low(2) = 1
    to_low(3) = -nzgrid
    to_low(4) = vmu_lo%llim_proc

    to_high(1) = vmu_lo%naky
    to_high(2) = vmu_lo%nakx
    to_high(3) = vmu_lo%nzed
    to_high(4) = vmu_lo%ulim_alloc

    call set_redist_character_type (kxkyz2vmu, 'kxkyz2vmu')

    call init_redist (kxkyz2vmu, 'c', to_low, to_high, to_list, &
         from_low, from_high, from_list)

    call delete_list (to_list)
    call delete_list (from_list)

  end subroutine init_kxkyz_to_vmu_redistribute

  subroutine init_kxyz_to_vmu_redistribute

    use mp, only: nproc
    use stella_layouts, only: kxyz_lo, vmu_lo
    use stella_layouts, only: kxyzidx2vmuidx
    use stella_layouts, only: idx_local, proc_id
    use redistribute, only: index_list_type, init_redist
    use redistribute, only: delete_list, set_redist_character_type
    use vpamu_grids, only: nvgrid, nmu
    use zgrid, only: nzgrid

    implicit none

    type (index_list_type), dimension (0:nproc-1) :: to_list, from_list
    integer, dimension (0:nproc-1) :: nn_to, nn_from
    integer, dimension (3) :: from_low, from_high
    integer, dimension (4) :: to_high, to_low
    integer :: ikxyz, ivmu
    integer :: iv, imu, iy, ikx, ig
    integer :: ip, n
    logical :: initialized = .false.

    if (initialized) return
    initialized = .true.

    ! count number of elements to be redistributed to/from each processor
    nn_to = 0
    nn_from = 0
    do ikxyz = kxyz_lo%llim_world, kxyz_lo%ulim_world
       do imu = 1, nmu
          do iv = -nvgrid, nvgrid
             call kxyzidx2vmuidx (iv, imu, ikxyz, kxyz_lo, vmu_lo, iy, ikx, ig, ivmu)
             if (idx_local(kxyz_lo,ikxyz)) &
                  nn_from(proc_id(vmu_lo,ivmu)) = nn_from(proc_id(vmu_lo,ivmu)) + 1
             if (idx_local(vmu_lo,ivmu)) &
                  nn_to(proc_id(kxyz_lo,ikxyz)) = nn_to(proc_id(kxyz_lo,ikxyz)) + 1
          end do
       end do
    end do
    
    do ip = 0, nproc-1
       if (nn_from(ip) > 0) then
          allocate (from_list(ip)%first(nn_from(ip)))
          allocate (from_list(ip)%second(nn_from(ip)))
          allocate (from_list(ip)%third(nn_from(ip)))
       end if
       if (nn_to(ip) > 0) then
          allocate (to_list(ip)%first(nn_to(ip)))
          allocate (to_list(ip)%second(nn_to(ip)))
          allocate (to_list(ip)%third(nn_to(ip)))
          allocate (to_list(ip)%fourth(nn_to(ip)))
       end if
    end do

    ! get local indices of elements distributed to/from other processors
    nn_to = 0
    nn_from = 0

    ! loop over all vmu indices, find corresponding y indices
    do ikxyz = kxyz_lo%llim_world, kxyz_lo%ulim_world
       do imu = 1, nmu
          do iv = -nvgrid, nvgrid
             ! obtain corresponding y indices
             call kxyzidx2vmuidx (iv, imu, ikxyz, kxyz_lo, vmu_lo, iy, ikx, ig, ivmu)
             ! if vmu index local, set:
             ! ip = corresponding y processor
             ! from_list%first-third arrays = iv,imu,ikxyz  (ie vmu indices)
             ! later will send from_list to proc ip
             if (idx_local(kxyz_lo,ikxyz)) then
                ip = proc_id(vmu_lo,ivmu)
                n = nn_from(ip) + 1
                nn_from(ip) = n
                from_list(ip)%first(n) = iv
                from_list(ip)%second(n) = imu
                from_list(ip)%third(n) = ikxyz
             end if
             ! if y index local, set ip to corresponding vmu processor
             ! set to_list%first,second arrays = iy,iy  (ie y indices)
             ! will receive to_list from ip
             if (idx_local(vmu_lo,ivmu)) then
                ip = proc_id(kxyz_lo,ikxyz)
                n = nn_to(ip) + 1
                nn_to(ip) = n
                to_list(ip)%first(n) = iy
                to_list(ip)%second(n) = ikx
                to_list(ip)%third(n) = ig
                to_list(ip)%fourth(n) = ivmu
             end if
          end do
       end do
    end do

    from_low(1) = -nvgrid
    from_low(2) = 1
    from_low(3) = kxyz_lo%llim_proc

    from_high(1) = nvgrid
    from_high(2) = nmu
    from_high(3) = kxyz_lo%ulim_alloc

    to_low(1) = 1
    to_low(2) = 1
    to_low(3) = -nzgrid
    to_low(4) = vmu_lo%llim_proc

    to_high(1) = vmu_lo%ny
    to_high(2) = vmu_lo%nakx
    to_high(3) = vmu_lo%nzed
    to_high(4) = vmu_lo%ulim_alloc

    call set_redist_character_type (kxyz2vmu, 'kxyz2vmu')

    call init_redist (kxyz2vmu, 'c', to_low, to_high, to_list, &
         from_low, from_high, from_list)

    call delete_list (to_list)
    call delete_list (from_list)

  end subroutine init_kxyz_to_vmu_redistribute

  subroutine init_cfl
    
    use mp, only: proc0, nproc, max_allreduce
    use dist_fn_arrays, only: wdriftx, wdrifty
    use stella_time, only: cfl_dt, code_dt, write_dt
    use zgrid, only: delzed
    use vpamu_grids, only: dvpa
    use kt_grids, only: akx, aky

    implicit none
    
    real :: cfl_dt_mirror, cfl_dt_stream
    real :: cfl_dt_wdriftx, cfl_dt_wdrifty
    real :: zero
    real :: wdriftx_max, wdrifty_max

    ! avoid divide by zero in cfl_dt terms below
    zero = 100.*epsilon(0.)

    ! FLAG -- assuming equal spacing in zed!

    ! get the local max value of wdriftx on each processor
    wdriftx_max = maxval(abs(wdriftx))
    ! compare these max values across processors to get global max
    if (nproc > 1) call max_allreduce (wdriftx_max)
    cfl_dt_wdriftx = code_dt/max(maxval(akx)*wdriftx_max,zero)
    cfl_dt = cfl_dt_wdriftx

    if (stream_explicit) then
       cfl_dt_stream = code_dt*delzed(0)/max(maxval(abs(stream*2.0)),zero)
       cfl_dt = min(cfl_dt,cfl_dt_stream)
    end if

    if (mirror_explicit) then
       cfl_dt_mirror = code_dt*dvpa/max(maxval(abs(mirror*2.0)),zero)
       cfl_dt = min(cfl_dt,cfl_dt_mirror)
    end if

    if (wdrifty_explicit) then
       ! get the local max value of wdrifty on each processor
       wdrifty_max = maxval(abs(wdrifty))
       ! compare these max values across processors to get global max
       if (nproc > 1) call max_allreduce (wdrifty_max)
       cfl_dt_wdrifty = code_dt/max(maxval(abs(aky))*wdrifty_max,zero)
       cfl_dt = min(cfl_dt,cfl_dt_wdrifty)
    end if
    
    if (proc0) then
       write (*,'(a16)') 'LINEAR CFL_DT: '
       write (*,'(a9,e12.4)') 'wdriftx: ', cfl_dt_wdriftx
       if (wdrifty_explicit) write (*,'(a9,e12.4)') 'wdrifty: ', cfl_dt_wdrifty
       if (stream_explicit) write (*,'(a9,e12.4)'), 'stream: ', cfl_dt_stream
       if (mirror_explicit) write (*,'(a9,e12.4)'), 'mirror: ', cfl_dt_mirror
       write (*,*)
    end if

    if (code_dt > cfl_dt) then
       if (proc0) then
          write (*,'(a21,e12.4,a32)') 'user-specified delt =', code_dt, 'does not satisfy CFL condition.'
          write (*,'(a32,e12.4)') 'changing code_dt to the CFL dt =', cfl_dt
       end if
       code_dt = cfl_dt
       call reset_dt
    else if (proc0) then
       call write_dt
       write (*,*)
    end if
    
  end subroutine init_cfl

  subroutine reset_dt

    implicit none

    ! need to recompute mirror and streaming terms
    ! to account for updated code_dt
    mirrorinit = .false.
    streaminit = .false.
    wdriftinit = .false.
    wstarinit = .false.
    call init_wstar
    call init_wdrift
    call init_mirror
    call init_parstream
    if (wstar_implicit) then
       get_fields_wstar_initialized = .false.
       ! call to init_get_fields only needed for initial step
       call init_get_fields
       call init_get_fields_wstar
    end if

  end subroutine reset_dt

  subroutine advance_stella

    use dist_fn_arrays, only: gold, gnew
    use fields_arrays, only: phi, apar

    implicit none

    ! advance the explicit parts of the GKE
    if (.not.fully_implicit) then
       if (explicit_rk4) then
          call advance_explicit_rk4 (gold, gnew)
       else
          call advance_explicit (gold, gnew)
       end if
    end if

    ! use operator splitting to seprately evolve
    ! all terms treated implicitly
    if (.not.fully_explicit) call advance_implicit (phi, apar, gnew)

    gold = gnew

  end subroutine advance_stella

  subroutine advance_explicit (gold, gnew)

    use dist_fn_arrays, only: g1, g2
    use zgrid, only: nzgrid
    use stella_layouts, only: kxkyz_lo

    implicit none

    complex, dimension (:,:,-nzgrid:,kxkyz_lo%llim_proc:), intent (in out) :: gold
    complex, dimension (:,:,-nzgrid:,kxkyz_lo%llim_proc:), intent (out) :: gnew

    integer :: icnt
    logical :: restart_time_step

    ! if CFL condition is violated by nonlinear term
    ! then must modify time step size and restart time step
    ! assume false and test
    restart_time_step = .false.

    icnt = 1
    ! SSP rk3 algorithm to advance explicit part of code
    ! if GK equation written as dg/dt = rhs - vpar . grad h,
    ! solve_gke returns rhs*dt
    do while (icnt <= 3)
       select case (icnt)
       case (1)
          call solve_gke (gold, g1, restart_time_step)
       case (2)
          g1 = gold + g1
          call solve_gke (g1, g2, restart_time_step)
       case (3)
          g2 = g1 + g2
          call solve_gke (g2, gnew, restart_time_step)
       end select
       if (restart_time_step) then
          icnt = 1
       else
          icnt = icnt + 1
       end if
    end do

    ! this is gbar at intermediate time level
    gnew = gold/3. + 0.5*g1 + (g2 + gnew)/6.

  end subroutine advance_explicit

  subroutine advance_explicit_rk4 (gold, gnew)

    use dist_fn_arrays, only: g1, g2, g3
    use zgrid, only: nzgrid
    use stella_layouts, only: kxkyz_lo

    implicit none

    complex, dimension (:,:,-nzgrid:,kxkyz_lo%llim_proc:), intent (in out) :: gold
    complex, dimension (:,:,-nzgrid:,kxkyz_lo%llim_proc:), intent (out) :: gnew

    integer :: icnt
    logical :: restart_time_step

    ! if CFL condition is violated by nonlinear term
    ! then must modify time step size and restart time step
    ! assume false and test
    restart_time_step = .false.

    icnt = 1
    ! SSP rk3 algorithm to advance explicit part of code
    ! if GK equation written as dg/dt = rhs - vpar . grad h,
    ! solve_gke returns rhs*dt
    do while (icnt <= 4)
       select case (icnt)
       case (1)
          call solve_gke (gold, g1, restart_time_step)
       case (2)
          ! g1 is h*k1
          g3 = gold + 0.5*g1
          call solve_gke (g3, g2, restart_time_step)
          g1 = g1 + 2.*g2
       case (3)
          ! g2 is h*k2
          g2 = gold+0.5*g2
          call solve_gke (g2, g3, restart_time_step)
          g1 = g1 + 2.*g3
       case (4)
          ! g3 is h*k3
          g3 = gold+g3
          call solve_gke (g3, gnew, restart_time_step)
          g1 = g1 + gnew
       end select
       if (restart_time_step) then
          icnt = 1
       else
          icnt = icnt + 1
       end if
    end do

    ! this is gbar at intermediate time level
    gnew = gold + g1/6.

  end subroutine advance_explicit_rk4

  subroutine solve_gke (gin, rhs_ky, restart_time_step)

    use job_manage, only: time_message
    use dist_fn_arrays, only: gbar_to_h, g_to_h
    use dist_fn_arrays, only: gvmu
    use fields_arrays, only: phi, apar
    use stella_layouts, only: vmu_lo
    use stella_transforms, only: transform_y2ky
    use redistribute, only: gather, scatter
    use run_parameters, only: fphi, fapar
    use run_parameters, only: nonlinear
    use zgrid, only: nzgrid
    use vpamu_grids, only: nvgrid, nmu
    use kt_grids, only: nakx, ny
    use kt_grids, only: alpha_global

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gin
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (out), target :: rhs_ky
    logical, intent (out) :: restart_time_step

    complex, dimension (:,:,:,:), allocatable, target :: rhs_y
    complex, dimension (:,:,:,:), pointer :: rhs

    rhs_ky = 0.

    if (alpha_global) then
       allocate (rhs_y(ny,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_proc))
       rhs_y = 0.
       rhs => rhs_y
    else
       rhs => rhs_ky
    end if

    ! start with gbar in k-space and (ky,kx,z) local

    ! obtain fields corresponding to gbar
!    call advance_fields (gin, phi, apar, h_in=.false.)
    call advance_fields (gin, phi, apar, dist='gbar')

    ! calculate and add ExB nonlinearity to RHS of GK eqn
    ! do this first, as the CFL condition may require a change in time step
    ! and thus recomputation of mirror, wdrift, wstar, and parstream
    if (nonlinear) then
       ! convert from h to g for nonlinearity
!       call g_to_h (gin, phi, -fphi)
       call advance_ExB_nonlinearity (gin, rhs, restart_time_step)
       ! finished with nonlinearity, so convert back to h
!       call g_to_h (gin, phi, fphi)
    else
       restart_time_step = .false.
    end if

    ! switch from g = h + (Ze/T)*<chi>*F_0 to h = f + (Ze/T)*phi*F_0
    call gbar_to_h (gin, phi, apar, fphi, fapar)
    call gbar_to_h (gvmu, phi, apar, fphi, fapar)

    if (.not.restart_time_step) then
       ! calculate and add mirror term to RHS of GK eqn
       if (mirror_explicit) call advance_mirror_explicit (gin, rhs)
       ! calculate and add alpha-component of magnetic drift term to RHS of GK eqn
       if (wdrifty_explicit) call advance_wdrifty_explicit (gin, rhs)
       ! calculate and add psi-component of magnetic drift term to RHS of GK eqn
       call advance_wdriftx (gin, rhs)
       
       if (alpha_global) then
          call transform_y2ky (rhs_y, rhs_ky)
          deallocate (rhs_y)
       end if
       
       ! calculate and add parallel streaming term to RHS of GK eqn
       if (stream_explicit) call advance_parallel_streaming_explicit (gin, rhs_ky)
       ! calculate and add omega_* term to RHS of GK eqn
       if (wstar_explicit) call advance_wstar_explicit (rhs_ky)
       ! calculate and add collision term to RHS of GK eqn
       !    call advance_collisions
    end if

    ! switch from h back to gbar
    call gbar_to_h (gin, phi, apar, -fphi, -fapar)

    nullify (rhs)

  end subroutine solve_gke

  subroutine advance_fields (g, phi, apar, dist)

    use mp, only: proc0
    use stella_layouts, only: vmu_lo
    use job_manage, only: time_message
    use redistribute, only: scatter
    use dist_fn_arrays, only: gvmu
!    use fields_arrays, only: phi, apar
    use zgrid, only: nzgrid

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:), intent (out) :: phi, apar
    character (*), intent (in) :: dist

    ! time the communications + field solve
    if (proc0) call time_message(.false.,time_gke(:,1),' fields')
    ! first gather (vpa,mu) onto processor for v-space operations
    ! v-space operations are field solve, dg/dvpa, and collisions
    if (debug) write (*,*) 'dist_fn::advance_stella::scatter'
    call scatter (kxkyz2vmu, g, gvmu)
    ! given gvmu with vpa and mu local, calculate the corresponding fields
    if (debug) write (*,*) 'dist_fn::advance_stella::get_fields'
    call get_fields (gvmu, phi, apar, dist)
    ! time the communications + field solve
    if (proc0) call time_message(.false.,time_gke(:,1),' fields')

  end subroutine advance_fields

  subroutine advance_parallel_streaming_explicit (g, gout)

    use mp, only: proc0
    use stella_layouts, only: vmu_lo
    use job_manage, only: time_message
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gout

    complex, dimension (:,:,:,:), allocatable :: g0

    allocate (g0(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    ! parallel streaming stays in ky,kx,z space with ky,kx,z local
    if (proc0) call time_message(.false.,time_gke(:,3),' Stream advance')
    ! get dg/dz, with z the parallel coordinate and store in g0
    if (debug) write (*,*) 'dist_fn::solve_gke::get_dgdz'
    call get_dgdz (g, g0)
    ! multiply dg/dz with vpa*(b . grad z) and add to source (RHS of GK equation)
    if (debug) write (*,*) 'dist_fn::solve_gke::add_stream_term'
    call add_stream_term (g0, gout)
    if (proc0) call time_message(.false.,time_gke(:,3),' Stream advance')
    deallocate (g0)

  end subroutine advance_parallel_streaming_explicit

  subroutine advance_wstar_explicit (gout)

    use mp, only: proc0
    use job_manage, only: time_message
    use fields_arrays, only: phi, apar
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gout

    complex, dimension (:,:,:,:), allocatable :: g0

    allocate (g0(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    ! omega_* stays in ky,kx,z space with ky,kx,z local
    if (proc0) call time_message(.false.,time_gke(:,6),' wstar advance')
    ! get d<chi>/dy
    if (debug) write (*,*) 'dist_fn::solve_gke::get_dchidy'
    call get_dchidy (phi, apar, g0)
    ! multiply with omega_* coefficient and add to source (RHS of GK eqn)
    if (debug) write (*,*) 'dist_fn::solve_gke::add_wstar_term'
    call add_wstar_term (g0, gout)
    if (proc0) call time_message(.false.,time_gke(:,6),' wstar advance')
    deallocate (g0)

  end subroutine advance_wstar_explicit

  subroutine advance_mirror_explicit (g, gout)

    use mp, only: proc0
    use redistribute, only: gather, scatter
    use dist_fn_arrays, only: gvmu
    use job_manage, only: time_message
    use stella_layouts, only: kxyz_lo, kxkyz_lo, vmu_lo
    use stella_transforms, only: transform_ky2y
    use zgrid, only: nzgrid
    use kt_grids, only: alpha_global
    use kt_grids, only: nakx, naky, ny
    use vpamu_grids, only: nvgrid, nmu
    use vpamu_grids, only: vpa

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gout

    complex, dimension (:,:,:), allocatable :: g0v
    complex, dimension (:,:,:,:), allocatable :: g0x

    integer :: iv

    if (proc0) call time_message(.false.,time_gke(:,2),' Mirror advance')
    ! the mirror term is most complicated of all when doing full flux surface
    if (alpha_global) then
       allocate (g0v(-nvgrid:nvgrid,nmu,kxyz_lo%llim_proc:kxyz_lo%ulim_alloc))
       allocate (g0x(ny,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))

       ! for upwinding, need to evaluate dh/dvpa in y-space
       ! first must take h(ky) and transform to h(y)
       call transform_ky2y (g, g0x)
       ! second, remap h so velocities are local
       call scatter (kxyz2vmu, g0x, g0v)
       ! next, calculate dh/dvpa
       call get_dgdvpa_global (g0v)
       ! add in extra term coming from definition of h as h*exp(mu*B/T)

       ! FLAG -- NEED TO REPLACE GVMU BELOW !!!
       do iv = -nvgrid, nvgrid
          g0v(iv,:,:) = g0v(iv,:,:) + 2.0*vpa(iv)*gvmu(iv,:,:)
       end do


       ! then take the results and remap again so y,kx,z local.
       call gather (kxyz2vmu, g0v, g0x)
       ! finally add the mirror term to the RHS of the GK eqn
       call add_mirror_term_global (g0x, gout)
    else
       allocate (g0v(-nvgrid:nvgrid,nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
       allocate (g0x(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))

       ! get dg/dvpa and store in g0_vmu
       if (debug) write (*,*) 'dist_fn::advance_stella::get_dgdvpa'
!       call get_dgdvpa (gvmu, g0v)
       g0v = gvmu
       call get_dgdvpa_explicit (g0v)
       ! add in extra term coming from definition of h as h*exp(mu*B/T)
       do iv = -nvgrid, nvgrid
          g0v(iv,:,:) = g0v(iv,:,:) + 2.0*vpa(iv)*gvmu(iv,:,:)
       end do
       if (debug) write (*,*) 'dist_fn::advance_stella::gather'
       ! swap layouts so that (z,kx,ky) are local
       call gather (kxkyz2vmu, g0v, g0x)
       ! get mirror term and add to source
       call add_mirror_term (g0x, gout)
    end if
    deallocate (g0x, g0v)
    if (proc0) call time_message(.false.,time_gke(:,2),' Mirror advance')

  end subroutine advance_mirror_explicit

  subroutine advance_wdrifty_explicit (g, gout)

    use mp, only: proc0
    use stella_layouts, only: vmu_lo
    use job_manage, only: time_message
    use stella_transforms, only: transform_ky2y
    use zgrid, only: nzgrid
    use kt_grids, only: nakx, naky, ny
    use kt_grids, only: alpha_global

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gout

    complex, dimension (:,:,:,:), allocatable :: g0k, g0y

    ! alpha-component of magnetic drift (requires ky -> y)
    if (proc0) call time_message(.false.,time_gke(:,4),' dgdy advance')

    allocate (g0k(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    if (debug) write (*,*) 'dist_fn::solve_gke::get_dgdy'
    call get_dgdy (g, g0k)
    if (alpha_global) then
       allocate (g0y(ny,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
       call transform_ky2y (g0k, g0y)
       call add_dgdy_term_global (g0y, gout)
       deallocate (g0y)
    else
       if (debug) write (*,*) 'dist_fn::solve_gke::add_dgdy_term'
       call add_dgdy_term (g0k, gout)
    end if
    deallocate (g0k)

    if (proc0) call time_message(.false.,time_gke(:,4),' dgdy advance')

  end subroutine advance_wdrifty_explicit

  subroutine advance_wdriftx (g, gout)

    use mp, only: proc0
    use stella_layouts, only: vmu_lo
    use job_manage, only: time_message
    use stella_transforms, only: transform_ky2y
    use zgrid, only: nzgrid
    use kt_grids, only: nakx, naky, ny
    use kt_grids, only: alpha_global

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gout

    complex, dimension (:,:,:,:), allocatable :: g0k, g0y

    ! psi-component of magnetic drift (requires ky -> y)
    if (proc0) call time_message(.false.,time_gke(:,5),' dgdx advance')

    allocate (g0k(naky,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
    if (debug) write (*,*) 'dist_fn::solve_gke::get_dgdx'
    call get_dgdx (g, g0k)

    if (alpha_global) then
       allocate (g0y(ny,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
       call transform_ky2y (g0k, g0y)
       call add_dgdx_term_global (g0y, gout)
       deallocate (g0y)
    else
       if (debug) write (*,*) 'dist_fn::solve_gke::add_dgdx_term'
       call add_dgdx_term (g0k, gout)
    end if
    deallocate (g0k)
    if (proc0) call time_message(.false.,time_gke(:,5),' dgdx advance')

  end subroutine advance_wdriftx

  subroutine advance_ExB_nonlinearity (g, gout, restart_time_step)

    use mp, only: proc0, min_allreduce
    use stella_layouts, only: vmu_lo
    use job_manage, only: time_message
    use fields_arrays, only: phi, apar
    use stella_transforms, only: transform_ky2y, transform_y2ky
    use stella_transforms, only: transform_kx2x, transform_x2kx
    use stella_time, only: cfl_dt, code_dt
    use run_parameters, only: cfl_cushion
    use zgrid, only: nzgrid
    use kt_grids, only: nakx, naky, nx, ny
    use kt_grids, only: akx, aky
    use kt_grids, only: alpha_global, iky_max

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: gout
    logical, intent (out) :: restart_time_step
    
    integer :: ivmu, iz

    complex, dimension (:,:), allocatable :: g0k, g0kxy
    real, dimension (:,:), allocatable :: g0xy, g1xy, bracket

    ! alpha-component of magnetic drift (requires ky -> y)
    if (proc0) call time_message(.false.,time_gke(:,7),' ExB nonlinear advance')

    restart_time_step = .false.

    allocate (g0k(naky,nakx))
    allocate (g0kxy(ny,nakx))
    allocate (g0xy(ny,nx))
    allocate (g1xy(ny,nx))
    allocate (bracket(ny,nx))

    ! FLAG -- NEED TO ADD IN EQUIVALENT OF KXFAC!!!
    ! something like q/rho/drhodpsin, possibly with factor of 1/2

    if (debug) write (*,*) 'dist_fn::solve_gke::advance_ExB_nonlinearity::get_dgdy'
    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       do iz = -nzgrid, nzgrid
          call get_dgdy (g(:,:,iz,ivmu), g0k)
          call transform_ky2y (g0k, g0kxy)
          call transform_kx2x (g0kxy, g0xy)
          call get_dchidx (iz, ivmu, phi(:,:,iz), apar(:,:,iz), g0k)
          call transform_ky2y (g0k, g0kxy)
          call transform_kx2x (g0kxy, g1xy)
          g1xy = g1xy*nonlin_fac
          bracket = -g0xy*g1xy
          cfl_dt = min(cfl_dt,1/(maxval(abs(g1xy))*aky(iky_max)))

          call get_dgdx (g(:,:,iz,ivmu), g0k)
          call transform_ky2y (g0k, g0kxy)
          call transform_kx2x (g0kxy, g0xy)
          call get_dchidy (iz, ivmu, phi(:,:,iz), apar(:,:,iz), g0k)
          call transform_ky2y (g0k, g0kxy)
          call transform_kx2x (g0kxy, g1xy)
          g1xy = g1xy*nonlin_fac
          bracket = bracket + g0xy*g1xy
          cfl_dt = min(cfl_dt,1/(maxval(abs(g1xy))*akx(nakx)))

          call transform_x2kx (bracket, g0kxy)
          if (alpha_global) then
             gout(:,:,iz,ivmu) = g0kxy
          else
             call transform_y2ky (g0kxy, gout(:,:,iz,ivmu))
          end if
       end do
    end do
    deallocate (g0k, g0kxy, g0xy, g1xy, bracket)

    call min_allreduce (cfl_dt)

    if (code_dt > cfl_dt) then
       if (proc0) then
          write (*,*) 'code_dt= ', code_dt, 'larger than cfl_dt= ', cfl_dt
          write (*,*) 'setting code_dt=cfl_dt and restarting time step'
       end if
       code_dt = cfl_dt
       call reset_dt
       restart_time_step = .true.
    else if (code_dt < cfl_dt*cfl_cushion) then
       if (proc0) then
          write (*,*) 'code_dt= ', code_dt, 'smaller than cfl_dt*cfl_cushion= ', cfl_dt*cfl_cushion
          write (*,*) 'setting code_dt=cfl_dt*cfl_cushion and restarting time step'
       end if
       code_dt = cfl_dt*cfl_cushion
       call reset_dt
    else
       gout = code_dt*gout
    end if

    if (proc0) call time_message(.false.,time_gke(:,7),' ExB nonlinear advance')

  end subroutine advance_ExB_nonlinearity

!   subroutine get_dgdvpa (g)

!     use finite_differences, only: first_order_upwind
!     use stella_layouts, only: kxkyz_lo, iz_idx, is_idx
!     use vpamu_grids, only: nvgrid, nmu, dvpa

!     implicit none

!     complex, dimension (-nvgrid:,:,kxkyz_lo%llim_proc:), intent (in out) :: g

!     integer :: ikxkyz, imu, iz, is
!     complex, dimension (:), allocatable :: tmp

!     allocate (tmp(-nvgrid:nvgrid))

!     do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
!        iz = iz_idx(kxkyz_lo,ikxkyz)
!        is = is_idx(kxkyz_lo,ikxkyz)
!        do imu = 1, nmu
!           call first_order_upwind (-nvgrid,g(:,imu,ikxkyz),dvpa,mirror_sign(1,iz),tmp)
!           ! get h + mirror*dh/dvpa
!           g(:,imu,ikxkyz) = g(:,imu,ikxkyz) + mirror(1,iz,imu,is)*tmp
!        end do
!     end do

!     deallocate (tmp)

!   end subroutine get_dgdvpa

  subroutine get_dgdvpa_global (g)

    use finite_differences, only: third_order_upwind
    use stella_layouts, only: kxyz_lo, iz_idx, iy_idx, is_idx
    use vpamu_grids, only: nvgrid, nmu, dvpa

    implicit none

    complex, dimension (-nvgrid:,:,kxyz_lo%llim_proc:), intent (in out) :: g

    integer :: ikxyz, imu, iz, iy, is
    complex, dimension (:), allocatable :: tmp

    allocate (tmp(-nvgrid:nvgrid))
    do ikxyz = kxyz_lo%llim_proc, kxyz_lo%ulim_proc
       iz = iz_idx(kxyz_lo,ikxyz)
       iy = iy_idx(kxyz_lo,ikxyz)
       is = is_idx(kxyz_lo,ikxyz)
       do imu = 1, nmu
          ! tmp is dh/dvpa
          call third_order_upwind (-nvgrid,g(:,imu,ikxyz),dvpa,mirror_sign(iy,iz),tmp)
          ! get h - mirror*dh/dvpa
!          g(:,imu,ikxyz) = g(:,imu,ikxyz) + mirror(iy,iz,imu,is)*tmp
          g(:,imu,ikxyz) = tmp
       end do
    end do
    deallocate (tmp)

  end subroutine get_dgdvpa_global

  subroutine get_dgdvpa_explicit (g)

    use finite_differences, only: third_order_upwind
    use stella_layouts, only: kxkyz_lo, iz_idx, is_idx
    use vpamu_grids, only: nvgrid, nmu, dvpa

    implicit none

    complex, dimension (-nvgrid:,:,kxkyz_lo%llim_proc:), intent (in out) :: g

    integer :: ikxkyz, imu, iz, is
    complex, dimension (:), allocatable :: tmp

    allocate (tmp(-nvgrid:nvgrid))

    do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
       iz = iz_idx(kxkyz_lo,ikxkyz)
       is = is_idx(kxkyz_lo,ikxkyz)
       do imu = 1, nmu
          call third_order_upwind (-nvgrid,g(:,imu,ikxkyz),dvpa,mirror_sign(1,iz),tmp)
!          g(:,imu,ikxkyz) = g(:,imu,ikxkyz) + 2.0*mirror(1,iz,imu,is)*tmp
!          g(:,imu,ikxkyz) = 2.0*mirror(1,iz,imu,is)*tmp
          g(:,imu,ikxkyz) = tmp
       end do
    end do

    deallocate (tmp)

  end subroutine get_dgdvpa_explicit

  subroutine add_mirror_term (g, src)

    use stella_layouts, only: vmu_lo
    use stella_layouts, only: imu_idx, is_idx
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: imu, is, ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       imu = imu_idx(vmu_lo,ivmu)
       is = is_idx(vmu_lo,ivmu)
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(spread(2.0*mirror(1,:,imu,is),1,naky),2,nakx)*g(:,:,:,ivmu)
!       src(:,:,:,ivmu) = src(:,:,:,ivmu) + g(:,:,:,ivmu)
    end do

  end subroutine add_mirror_term

  subroutine add_mirror_term_global (g, src)

    use stella_layouts, only: vmu_lo
    use stella_layouts, only: imu_idx, is_idx
    use zgrid, only: nzgrid
    use kt_grids, only: nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: imu, is, ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       imu = imu_idx(vmu_lo,ivmu)
       is = is_idx(vmu_lo,ivmu)
!       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(mirror(:,:,imu,is),2,nakx)*g(:,:,:,ivmu)
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(2.0*mirror(:,:,imu,is),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_mirror_term_global

  subroutine get_dgdz (g, dgdz)

    use finite_differences, only: third_order_upwind_zed
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: iv_idx
    use zgrid, only: nzgrid, delzed
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (out) :: dgdz

    integer :: ivmu, iseg, ie, iky, iv
    complex, dimension (2) :: gleft, gright

    ! FLAG -- assuming delta zed is equally spaced below!
    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       iv = iv_idx(vmu_lo,ivmu)
       do iky = 1, naky
          do ie = 1, neigen(iky)
             do iseg = 1, nsegments(ie,iky)
                ! if iseg,ie,iky corresponds to negative kx, no need to solve
                ! as it will be constrained by reality condition
                if (ikxmod(iseg,ie,iky) > nakx) cycle
                ! first fill in ghost zones at boundaries in g(z)
                call fill_zed_ghost_zones (iseg, ie, iky, g(:,:,:,ivmu), gleft, gright)
                ! now get dg/dz
                call third_order_upwind_zed (ig_low(iseg), iseg, nsegments(ie,iky), &
                     g(iky,ikxmod(iseg,ie,iky),ig_low(iseg):ig_up(iseg),ivmu), &
                     delzed(0), stream_sign(iv), gleft, gright, &
                     dgdz(iky,ikxmod(iseg,ie,iky),ig_low(iseg):ig_up(iseg),ivmu))
             end do
          end do
       end do
    end do

  end subroutine get_dgdz

  subroutine add_stream_term (g, src)

    use stella_layouts, only: vmu_lo
    use stella_layouts, only: iv_idx, is_idx
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: iv, is, ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       iv = iv_idx(vmu_lo,ivmu)
       is = is_idx(vmu_lo,ivmu)
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + 2.0*spread(spread(stream(:,iv,is),1,naky),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_stream_term

  subroutine fill_zed_ghost_zones_real (iseg, ie, iky, g, gleft, gright)

    use zgrid, only: nzgrid
    use kt_grids, only: ntheta0, nakx, naky
    use kt_grids, only: aky

    implicit none

    integer, intent (in) :: iseg, ie, iky
    real, dimension (:,:,-nzgrid:), intent (in) :: g
    real, dimension (:), intent (out) :: gleft, gright

    integer :: ikxneg, ikyneg

    ! stream_sign > 0 --> stream speed < 0

    if (iseg == 1) then
       gleft = 0.0
    else
       ! if trying to connect to a segment that 
       ! corresponds to kx negative, use reality
       ! condition to connect to positive kx instead
       ! (with sign of ky flipped)
       if (ikxmod(iseg-1,ie,iky) > nakx) then
          ikxneg = ntheta0-ikxmod(iseg-1,ie,iky)+2
          if (abs(aky(iky)) < epsilon(0.)) then
             ikyneg = iky
          else
             ikyneg = naky-iky+2
          end if
          gleft = g(ikyneg,ikxneg,ig_up(iseg-1)-2:ig_up(iseg-1)-1)
       else
          gleft = g(iky,ikxmod(iseg-1,ie,iky),ig_up(iseg-1)-2:ig_up(iseg-1)-1)
       end if
    end if
    
    if (nsegments(ie,iky) > iseg) then
       ! if trying to connect to a segment that 
       ! corresponds to kx negative, use reality
       ! condition to connect to positive kx instead
       ! (with sign of ky flipped)
       if (ikxmod(iseg+1,ie,iky) > nakx) then
          ikxneg = ntheta0-ikxmod(iseg+1,ie,iky)+2
          if (abs(aky(iky)) < epsilon(0.)) then
             ikyneg = iky
          else
             ikyneg = naky-iky+2
          end if
          gright = g(ikyneg,ikxneg,ig_low(iseg+1)+1:ig_low(iseg+1)+2)
       else
          ! connect to segment with larger theta-theta0 (on right)
          gright = g(iky,ikxmod(iseg+1,ie,iky),ig_low(iseg+1)+1:ig_low(iseg+1)+2)
       end if
    else
       gright = 0.0
    end if
    
  end subroutine fill_zed_ghost_zones_real

  subroutine fill_zed_ghost_zones_complex (iseg, ie, iky, g, gleft, gright)

    use zgrid, only: nzgrid
    use kt_grids, only: ntheta0, nakx, naky
    use kt_grids, only: aky

    implicit none

    integer, intent (in) :: iseg, ie, iky
    complex, dimension (:,:,-nzgrid:), intent (in) :: g
    complex, dimension (:), intent (out) :: gleft, gright

    integer :: ikxneg, ikyneg

    ! stream_sign > 0 --> stream speed < 0

    if (iseg == 1) then
       gleft = 0.0
    else
       ! if trying to connect to a segment that 
       ! corresponds to kx negative, use reality
       ! condition to connect to positive kx instead
       ! (with sign of ky flipped)
       if (ikxmod(iseg-1,ie,iky) > nakx) then
          ikxneg = ntheta0-ikxmod(iseg-1,ie,iky)+2
          if (abs(aky(iky)) < epsilon(0.)) then
             ikyneg = iky
          else
             ikyneg = naky-iky+2
          end if
          gleft = conjg(g(ikyneg,ikxneg,ig_up(iseg-1)-2:ig_up(iseg-1)-1))
       else
          gleft = g(iky,ikxmod(iseg-1,ie,iky),ig_up(iseg-1)-2:ig_up(iseg-1)-1)
       end if
    end if
    
    if (nsegments(ie,iky) > iseg) then
       ! if trying to connect to a segment that 
       ! corresponds to kx negative, use reality
       ! condition to connect to positive kx instead
       ! (with sign of ky flipped)
       if (ikxmod(iseg+1,ie,iky) > nakx) then
          ikxneg = ntheta0-ikxmod(iseg+1,ie,iky)+2
          if (abs(aky(iky)) < epsilon(0.)) then
             ikyneg = iky
          else
             ikyneg = naky-iky+2
          end if
          gright = conjg(g(ikyneg,ikxneg,ig_low(iseg+1)+1:ig_low(iseg+1)+2))
       else
          ! connect to segment with larger theta-theta0 (on right)
          gright = g(iky,ikxmod(iseg+1,ie,iky),ig_low(iseg+1)+1:ig_low(iseg+1)+2)
       end if
    else
       gright = 0.0
    end if
    
  end subroutine fill_zed_ghost_zones_complex

  subroutine get_dgdy_4d (g, dgdy)

    use constants, only: zi
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: nakx, aky

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (out) :: dgdy

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       dgdy(:,:,:,ivmu) = zi*spread(spread(aky,2,nakx),3,2*nzgrid+1)*g(:,:,:,ivmu)
    end do

  end subroutine get_dgdy_4d

  subroutine get_dgdy_2d (g, dgdy)

    use constants, only: zi
    use kt_grids, only: nakx, aky

    implicit none

    complex, dimension (:,:), intent (in) :: g
    complex, dimension (:,:), intent (out) :: dgdy

    dgdy = zi*spread(aky,2,nakx)*g

  end subroutine get_dgdy_2d

  subroutine add_dgdy_term (g, src)

    use dist_fn_arrays, only: wdrifty
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(spread(wdrifty(1,:,ivmu),1,naky),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_dgdy_term

  subroutine add_dgdy_term_global (g, src)

    use dist_fn_arrays, only: wdrifty
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(wdrifty(:,:,ivmu),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_dgdy_term_global

  subroutine get_dgdx_4d (g, dgdx)

    use constants, only: zi
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: naky, akx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (out) :: dgdx

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       dgdx(:,:,:,ivmu) = zi*spread(spread(akx,1,naky),3,2*nzgrid+1)*g(:,:,:,ivmu)
    end do

  end subroutine get_dgdx_4d

  subroutine get_dgdx_2d (g, dgdx)

    use constants, only: zi
    use kt_grids, only: naky, akx

    implicit none

    complex, dimension (:,:), intent (in) :: g
    complex, dimension (:,:), intent (out) :: dgdx

    dgdx = zi*spread(akx,1,naky)*g

  end subroutine get_dgdx_2d

  subroutine add_dgdx_term (g, src)

    use dist_fn_arrays, only: wdriftx
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(spread(wdriftx(1,:,ivmu),1,naky),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_dgdx_term

  subroutine add_dgdx_term_global (g, src)

    use dist_fn_arrays, only: wdriftx
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + spread(wdriftx(:,:,ivmu),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_dgdx_term_global

  subroutine get_dchidy_4d (phi, apar, dchidy)

    use constants, only: zi
    use dist_fn_arrays, only: aj0x
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: is_idx, iv_idx
    use run_parameters, only: fphi, fapar
    use species, only: spec
    use zgrid, only: nzgrid
    use vpamu_grids, only: vpa
    use kt_grids, only: nakx, aky

    implicit none

    complex, dimension (:,:,-nzgrid:), intent (in) :: phi, apar
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (out) :: dchidy

    integer :: ivmu, iv, is

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       is = is_idx(vmu_lo,ivmu)
       iv = iv_idx(vmu_lo,ivmu)
       dchidy(:,:,:,ivmu) = zi*spread(spread(aky,2,nakx),3,2*nzgrid+1)*aj0x(:,:,:,ivmu) &
            * ( fphi*phi - fapar*vpa(iv)*spec(is)%stm*apar )
    end do

  end subroutine get_dchidy_4d

  subroutine get_dchidy_2d (iz, ivmu, phi, apar, dchidy)

    use constants, only: zi
    use dist_fn_arrays, only: aj0x
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: is_idx, iv_idx
    use run_parameters, only: fphi, fapar
    use species, only: spec
    use vpamu_grids, only: vpa
    use kt_grids, only: nakx, aky

    implicit none

    integer, intent (in) :: ivmu, iz
    complex, dimension (:,:), intent (in) :: phi, apar
    complex, dimension (:,:), intent (out) :: dchidy

    integer :: iv, is

    is = is_idx(vmu_lo,ivmu)
    iv = iv_idx(vmu_lo,ivmu)
    dchidy = zi*spread(aky,2,nakx)*aj0x(:,:,iz,ivmu) &
         * ( fphi*phi - fapar*vpa(iv)*spec(is)%stm*apar )
    
  end subroutine get_dchidy_2d

  subroutine get_dchidx (iz, ivmu, phi, apar, dchidx)

    use constants, only: zi
    use dist_fn_arrays, only: aj0x
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: is_idx, iv_idx
    use run_parameters, only: fphi, fapar
    use species, only: spec
    use vpamu_grids, only: vpa
    use kt_grids, only: akx, naky

    implicit none

    integer, intent (in) :: ivmu, iz
    complex, dimension (:,:), intent (in) :: phi, apar
    complex, dimension (:,:), intent (out) :: dchidx

    integer :: iv, is

    is = is_idx(vmu_lo,ivmu)
    iv = iv_idx(vmu_lo,ivmu)
    dchidx = zi*spread(akx,1,naky)*aj0x(:,:,iz,ivmu) &
         * ( fphi*phi - fapar*vpa(iv)*spec(is)%stm*apar )
    
  end subroutine get_dchidx

  subroutine add_wstar_term (g, src)

    use dist_fn_arrays, only: wstar
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in) :: g
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: src

    integer :: ivmu

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       src(:,:,:,ivmu) = src(:,:,:,ivmu) + 2.0*spread(spread(wstar(:,ivmu),1,naky),2,nakx)*g(:,:,:,ivmu)
    end do

  end subroutine add_wstar_term

  subroutine advance_implicit (phi, apar, g)

    use mp, only: mp_abort
    use run_parameters, only: fphi, fapar
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use dist_fn_arrays, only: gbar_to_h

    implicit none

    complex, dimension (:,:,-nzgrid:), intent (in out) :: phi, apar
    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g

    ! g^{*} (coming from explicit solve) is input
    ! get g^{**}, with g^{**}-g^{*} due to mirror term
    if (mirror_implicit) call advance_mirror_implicit (g, phi, apar)

    ! g^{**} is input
    ! get g^{***}, with g^{***}-g^{**} due to parallel streaming term
!    if (stream_implicit) call advance_parallel_streaming_implicit (g, phi, apar)

    if (wdrifty_implicit) call advance_wdrifty_implicit (g)
    if (wstar_implicit) call advance_wstar_implicit (g, phi, apar)

  end subroutine advance_implicit

  subroutine advance_mirror_implicit (g, phi, apar)

    use mp, only: proc0
    use job_manage, only: time_message
    use redistribute, only: gather, scatter
    use stella_layouts, only: vmu_lo, kxyz_lo, kxkyz_lo
    use stella_layouts, only: iy_idx, iz_idx
    use stella_transforms, only: transform_ky2y, transform_y2ky
    use run_parameters, only: fphi, fapar
    use zgrid, only: nzgrid
    use dist_fn_arrays, only: gbar_to_h
    use dist_fn_arrays, only: gvmu
    use kt_grids, only: alpha_global
    use kt_grids, only: ny, nakx
    use vpamu_grids, only: nvgrid, nmu

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g
    complex, dimension (:,:,-nzgrid:), intent (in out) :: phi, apar

    complex, dimension (:,:,:), allocatable :: g0v
    complex, dimension (:,:,:,:), allocatable :: g0x

    integer :: ikxyz, ikxkyz, iy, iz, imu

    if (proc0) call time_message(.false.,time_gke(:,2),' Mirror advance')

    ! now that we have h^{*}, need to solve
    ! h^{n+1} = h^{*} - dt*mu*bhat . grad B d((h^{n+1}+h^{*})/2)/dvpa
    ! define A_0^{-1} = dt*mu*bhat.gradB/2
    ! so that (A_0 + I)h^{n+1} = (A_0-I)h^{*}
    ! will need (I-A_0^{-1})h^{*} in Sherman-Morrison approach
    ! to invert and obtain h^{n+1}
    if (alpha_global) then
       allocate (g0v(-nvgrid:nvgrid,nmu,kxyz_lo%llim_proc:kxyz_lo%ulim_alloc))
       allocate (g0x(ny,nakx,-nzgrid:nzgrid,vmu_lo%llim_proc:vmu_lo%ulim_alloc))
       ! for upwinding, need to evaluate dh^{*}/dvpa in y-space
       ! first must take h^{*}(ky) and transform to h^{*}(y)
       call transform_ky2y (g, g0x)
       ! second, remap h so velocities are local
       call scatter (kxyz2vmu, g0x, g0v)
       ! next, take dh^{*}/dvpa and multiply with mirror coefficient in y-space.
       ! g= h^{*} is input and overwritten with (I-A_0^{-1})h^{*}
       call get_dgdvpa_global (g0v)
       ! invert_mirror_operator takes rhs of equation and
       ! returns h^{n+1}
       do ikxyz = kxyz_lo%llim_proc, kxyz_lo%ulim_proc
          iy = iy_idx(kxyz_lo,ikxyz)
          iz = iz_idx(kxyz_lo,ikxyz)
!          call invert_mirror_operator (ikxyz, mirror_sign(iy,iz),g0v(:,:,ikxyz))
       end do
       ! then take the results and remap again so y,kx,z local.
       call gather (kxyz2vmu, g0v, g0x)
       ! finally transform back from y to ky space
       call transform_y2ky (g0x, g)
       call scatter (kxyz2vmu, g, gvmu)
       ! get updated fields corresponding to h^{n+1}
       call get_fields (gvmu, phi, apar, dist='h')
    else
       ! get g^{*} with v-space on processor
       call scatter (kxkyz2vmu, g, gvmu)

       allocate (g0v(-nvgrid:nvgrid,nmu,kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
       allocate (g0x(1,1,1,1))

       g0v = gvmu
! BACKWARDS DIFFERENCE FLAG
!       call get_dgdvpa (g0v)
       ! invert_mirror_operator takes rhs of equation and
       ! returns g^{n+1}
       do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
          iz = iz_idx(kxkyz_lo,ikxkyz)
          do imu = 1, nmu
             call invert_mirror_operator (imu, ikxkyz, g0v(:,imu,ikxkyz))
          end do
       end do

       ! then take the results and remap again so ky,kx,z local.
       call gather (kxkyz2vmu, g0v, g)
    end if
    
    deallocate (g0x,g0v)

    if (proc0) call time_message(.false.,time_gke(:,2),' Mirror advance')

  end subroutine advance_mirror_implicit

  subroutine invert_mirror_operator (imu, ilo, g)

    use finite_differences, only: tridag
    use vpamu_grids, only: nvgrid

    implicit none

    integer, intent (in) :: imu, ilo
    complex, dimension (-nvgrid:), intent (in out) :: g

    call tridag (-nvgrid, mirror_tri_a(:,imu,ilo), mirror_tri_b(:,imu,ilo), mirror_tri_c(:,imu,ilo), g)

  end subroutine invert_mirror_operator

!   subroutine advance_parallel_streaming_implicit (g, phi, apar)

!     use mp, only: proc0, sum_allreduce
!     use stella_layouts, only: vmu_lo
!     use stella_layouts, only: iv_idx, imu_idx, is_idx
!     use job_manage, only: time_message
!     use run_parameters, only: fphi, fapar
!     use zgrid, only: nzgrid
!     use dist_fn_arrays, only: gbar_to_h

!     implicit none

!     complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g
!     complex, dimension (:,:,-nzgrid:), intent (in out) :: phi, apar

!     ! parallel streaming stays in ky,kx,z space with ky,kx,z local
!     if (proc0) call time_message(.false.,time_gke(:,3),' Stream advance')

! !    do iz = -nzgrid, nzgrid
! !       write (*,*) 'zeroth', iz, cabs(phi(1,1,iz)), sum(cabs(g(1,1,iz,:)))
! !    end do

!     ! input is g^{*}, output is g^{**}
!     ! where (1 + dt*(vpa . grad z)*d/dz) g^{**} = g^{*}
!     call invert_stream_operator (g)
!     ! next get fields corresponding to g^{**} (aka phi^{**})
!     call advance_fields (g, phi, apar, dist='gbar')

! !    do iz = -nzgrid, nzgrid
! !       write (*,*) 'first', iz, cabs(phi(1,1,iz)), sum(cabs(g(1,1,iz,:)))
! !    end do

!     ! get phi^{n+1} by operating on g^{n+1} equation with QN operator
!     ! input is phi^{**}, output is phi^{n+1}
!     call invert_stream_phi (phi)

! !    do iz = -nzgrid, nzgrid
! !       write (*,*) 'second', iz, cabs(phi(1,1,iz)), sum(cabs(g(1,1,iz,:)))
! !       write (*,*) 'second', iz, cabs(phi(1,1,iz)), cabs(g(1,1,iz,1))
! !    end do

!     ! solve for g^{n+1}, with g^{n+1} = g^{**} - dt*(vpa . grad z)*d(<phi^{n+1}>)/dz * Ze/T * F0
! !    ! note that the phi term modifies g in such a way that it does not change the associated field
! !    ! so phi^{n+1} = phi^{**}

!     call add_dphidz_term_to_stream (g, phi)

!     ! TMP FOR TESTING -- MAB
! !    call advance_fields (g, phi, apar, dist='gbar')

! !    do iz = -nzgrid, nzgrid
! !       write (*,*) 'third', iz, cabs(phi(1,1,iz)), sum(cabs(g(1,1,iz,:)))
! !       write (*,*) 'third', iz, cabs(phi(1,1,iz)), cabs(g(1,1,iz,1))
! !    end do
! !    stop

!     if (proc0) call time_message(.false.,time_gke(:,3),' Stream advance')

!   end subroutine advance_parallel_streaming_implicit

!   subroutine add_dphidz_term_to_stream (g, phi)

!     use finite_differences, only: second_order_centered_zed
!     use finite_differences, only: first_order_upwind_zed
!     use stella_layouts, only: vmu_lo
!     use stella_layouts, only: iv_idx, imu_idx, is_idx
!     use species, only: spec
!     use zgrid, only: nzgrid, delzed
!     use kt_grids, only: naky, nakx
!     use vpamu_grids, only: maxwellian
!     use dist_fn_arrays, only: aj0x

!     implicit none

!     complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g
!     complex, dimension (:,:,-nzgrid:), intent (in) :: phi

!     integer :: ivmu, iv, imu, is
!     integer :: iky, ie, iseg
!     integer :: ikx
!     complex, dimension (2) :: gleft, gright
!     real, dimension (2) :: j0left, j0right
!     real, dimension (:), allocatable :: gtmp1
!     complex, dimension (:), allocatable :: gtmp2, gtmp3
!     complex, dimension (:,:), allocatable :: gsave

!     allocate (gtmp1(-nzgrid:nzgrid))
!     allocate (gtmp2(-nzgrid:nzgrid))
!     allocate (gtmp3(-nzgrid:nzgrid))
!     allocate (gsave(nakx,-nzgrid:nzgrid))

!     do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
!        iv = iv_idx(vmu_lo,ivmu)
!        imu = imu_idx(vmu_lo,ivmu)
!        is = is_idx(vmu_lo,ivmu)
!        do iky = 1, naky
!           gsave = g(iky,:,:,ivmu)
!           do ie = 1, neigen(iky)
!              do iseg = 1, nsegments(ie,iky)
!                 ! if iseg,ie,iky corresponds to negative kx, no need to solve
!                 ! as it will be constrained by reality condition
!                 ikx = ikxmod(iseg,ie,iky)
!                 if (ikx > nakx) cycle

!                 ! first fill in ghost zones at boundaries in J0
!                 call fill_zed_ghost_zones (iseg, ie, iky, aj0x(:,:,:,ivmu), j0left, j0right)
!                 ! now get dJ0/dz
!                 call second_order_centered_zed (ig_low(iseg), iseg, nsegments(ie,iky), &
!                      aj0x(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu), &
!                      delzed(0), j0left, j0right, &
!                      gtmp1(ig_low(iseg):ig_up(iseg)))

!                 ! fill in ghost zones at boundaries in phi
!                 call fill_zed_ghost_zones (iseg, ie, iky, phi, gleft, gright)
!                 ! now get dphi/dz with upwinding
!                 call first_order_upwind_zed (ig_low(iseg), iseg, nsegments(ie,iky), &
!                      phi(iky,ikx,ig_low(iseg):ig_up(iseg)), &
!                      delzed(0), stream_sign(iv), gleft, gright, &
!                      gtmp2(ig_low(iseg):ig_up(iseg)))
!                 ! now get dphi/dz centered
!                 call second_order_centered_zed (ig_low(iseg), iseg, nsegments(ie,iky), &
!                      phi(iky,ikx,ig_low(iseg):ig_up(iseg)), &
!                      delzed(0), gleft, gright, &
!                      gtmp3(ig_low(iseg):ig_up(iseg)))

!                 ! take combination of upwinded and centered derivatives
!                 ! at boundaries, use only upwinded values
!                 if (iseg == 1 .and. stream_sign(iv) == 1) then
!                    gtmp2(ig_low(iseg)+1:) = stream_upwind*gtmp2(ig_low(iseg)+1:) &
!                         + (1.0-stream_upwind)*gtmp3(ig_low(iseg)+1:)
!                 else if (iseg == nsegments(ie,iky) .and. stream_sign(iv) == -1) then
!                    gtmp2(:ig_up(iseg)-1) = stream_upwind*gtmp2(:ig_up(iseg)-1) &
!                         + (1.0-stream_upwind)*gtmp3(:ig_up(iseg)-1)
!                 else
!                    gtmp2 = stream_upwind*gtmp2 + (1.0-stream_upwind)*gtmp3
!                 end if

!                 ! TMP FOR TESTING -- MAB
! !                gtmp1 = 0.

! !                if (ivmu == 1) then
! !                   do iz = ig_low(iseg), ig_up(iseg)
! !                      write (*,*) 'pre-g', iz, cabs(g(iky,ikx,iz,ivmu)), iky, ikx, ig_low(iseg), ig_up(iseg)
! !                   end do
!  !               end if

! !                g(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu) = g(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu) &
!                 g(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu) = gsave(ikx,ig_low(iseg):ig_up(iseg)) &
!                      + 2.0*stream(ig_low(iseg):ig_up(iseg),iv,is) &
!                      * (phi(iky,ikx,ig_low(iseg):ig_up(iseg))*gtmp1(ig_low(iseg):ig_up(iseg)) &
!                      + aj0x(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu)*gtmp2(ig_low(iseg):ig_up(iseg))) &
!                      * spec(is)%zt*anon(ig_low(iseg):ig_up(iseg),iv,imu)

! !                if (ivmu == 1) then
! !                   do iz = ig_low(iseg), ig_up(iseg)
! !                      write (*,*) 'gtmp2', iz, cabs(gtmp2(iz)), cabs(g(iky,ikx,iz,ivmu)), abs(gtmp1(iz)), cabs(gtmp3(iz)), stream(iz,iv,is), phi(iky,ikx,iz), aj0x(iky,ikx,iz,ivmu), anon(iz,iv,imu)
! !                   end do
! !                end if

!              end do
!           end do
!        end do
!     end do
    
!     deallocate (gtmp1, gtmp2, gtmp3)
!     deallocate (gsave)

!   end subroutine add_dphidz_term_to_stream

  subroutine invert_stream_operator (g)

    use stella_layouts, only: vmu_lo
    use stella_layouts, only: iv_idx, is_idx
    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx, ntheta0
    use kt_grids, only: aky

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g

    integer :: iky, ie, iseg, ivmu
    integer :: iv, is
    integer :: llim, ulim
    integer :: ikx, nz
    integer :: ikxneg, ikyneg
    complex, dimension (:), allocatable :: gext

    nz = maxval(ig_up-ig_low)

    allocate (gext(nz*maxval(nsegments)+1))

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       iv = iv_idx(vmu_lo,ivmu)
       is = is_idx(vmu_lo,ivmu)
       do iky = 1, naky
          if (abs(aky(iky)) < epsilon(0.)) then
             ikyneg = iky
          else
             ikyneg = naky-iky+2
          end if
          do ie = 1, neigen(iky)
             ! avoid double-counting at boundaries between 2pi segments
             iseg = 1
             ikx = ikxmod(iseg,ie,iky)
             llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
             if (ikx > nakx) then
                ! have to do something special here (connect to negative ky)
                ! in order to finish up the connections
                ikxneg = ntheta0-ikx+2
                gext(llim:ulim) = conjg(g(ikyneg,ikxneg,ig_low(iseg):ig_up(iseg),ivmu))
             else
                gext(llim:ulim) = g(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu)
             end if
             if (nsegments(ie,iky) > 1) then
                do iseg = 2, nsegments(ie,iky)
                   ikx = ikxmod(iseg,ie,iky)
                   llim = ulim+1
                   ulim = llim+ig_up(iseg)-ig_low(iseg)-1
                   if (ikx > nakx) then
                      ! have to do something special here (connect to negative ky)
                      ! in order to finish up the connections
                      ikxneg = ntheta0-ikx+2
                      gext(llim:ulim) = conjg(g(ikyneg,ikxneg,ig_low(iseg)+1:ig_up(iseg),ivmu))
                   else
                      gext(llim:ulim) = g(iky,ikx,ig_low(iseg)+1:ig_up(iseg),ivmu)
                   end if
                end do
             end if

! BACKWARDS DIFFERENCE FLAG
!             call get_dgdz_implicit (iky, ie, iv, is, gext(:ulim))

             call stream_tridiagonal_solve (iky, ie, iv, is, gext(:ulim))
             
             iseg = 1
             ikx = ikxmod(iseg,ie,iky)
             if (ikx <= nakx) then
                llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
                g(iky,ikx,ig_low(iseg):ig_up(iseg),ivmu) = gext(llim:ulim)
             end if
             if (nsegments(ie,iky) > 1) then
                do iseg = 2, nsegments(ie,iky)
                   ikx = ikxmod(iseg,ie,iky)
                   if (ikx <= nakx) then
                      llim = ulim+1
                      ulim = llim+ig_up(iseg)-ig_low(iseg)-1
                      g(iky,ikx,ig_low(iseg),ivmu) = gext(llim-1)
                      g(iky,ikx,ig_low(iseg)+1:ig_up(iseg),ivmu) = gext(llim:ulim)
                   end if
                end do
             end if
          end do
       end do
    end do
    
    deallocate (gext)
    
  end subroutine invert_stream_operator

  subroutine invert_stream_phi (phi)

    use zgrid, only: nzgrid
    use kt_grids, only: naky, nakx, ntheta0
    use kt_grids, only: aky

    implicit none

    complex, dimension (:,:,-nzgrid:), intent (in out) :: phi

    integer :: iky, ie, iseg
    integer :: llim, ulim
    integer :: ikx, nz
    integer :: ikxneg, ikyneg
    complex, dimension (:), allocatable :: phiext

    nz = maxval(ig_up-ig_low)

    allocate (phiext(nz*maxval(nsegments)+1))

    do iky = 1, naky
       if (abs(aky(iky)) < epsilon(0.)) then
          ikyneg = iky
       else
          ikyneg = naky-iky+2
       end if
       do ie = 1, neigen(iky)
          ! avoid double-counting at boundaries between 2pi segments
          iseg = 1
          ikx = ikxmod(iseg,ie,iky)
          llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
          if (ikx > nakx) then
             ! have to do something special here (connect to negative ky)
             ! in order to finish up the connections
             ikxneg = ntheta0-ikx+2
             phiext(llim:ulim) = conjg(phi(ikyneg,ikxneg,ig_low(iseg):ig_up(iseg)))
          else
             phiext(llim:ulim) = phi(iky,ikx,ig_low(iseg):ig_up(iseg))
          end if
          if (nsegments(ie,iky) > 1) then
             do iseg = 2, nsegments(ie,iky)
                ikx = ikxmod(iseg,ie,iky)
                llim = ulim+1
                ulim = llim+ig_up(iseg)-ig_low(iseg)-1
                if (ikx > nakx) then
                   ! have to do something special here (connect to negative ky)
                   ! in order to finish up the connections
                   ikxneg = ntheta0-ikx+2
                   phiext(llim:ulim) = conjg(phi(ikyneg,ikxneg,ig_low(iseg)+1:ig_up(iseg)))
                else
                   phiext(llim:ulim) = phi(iky,ikx,ig_low(iseg)+1:ig_up(iseg))
                end if
             end do
          end if

! BACKWARDS DIFFERENCE FLAG
!             call get_dgdz_implicit (iky, ie, iv, is, gext(:ulim))

          call stream_phi_tridiagonal_solve (iky, ie, phiext(:ulim))
             
          iseg = 1
          ikx = ikxmod(iseg,ie,iky)
          if (ikx <= nakx) then
             llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
             phi(iky,ikx,ig_low(iseg):ig_up(iseg)) = phiext(llim:ulim)
          end if
          if (nsegments(ie,iky) > 1) then
             do iseg = 2, nsegments(ie,iky)
                ikx = ikxmod(iseg,ie,iky)
                if (ikx <= nakx) then
                   llim = ulim+1
                   ulim = llim+ig_up(iseg)-ig_low(iseg)-1
                   phi(iky,ikx,ig_low(iseg)) = phiext(llim-1)
                   phi(iky,ikx,ig_low(iseg)+1:ig_up(iseg)) = phiext(llim:ulim)
                end if
             end do
          end if
       end do
    end do
    
    deallocate (phiext)
    
  end subroutine invert_stream_phi

!   subroutine get_dgdz_implicit (iky, ie, iv, is, g)

!     use finite_differences, only: first_order_upwind
!     use zgrid, only: delzed

!     implicit none

!     integer, intent (in) :: iv, is, ie, iky
!     complex, dimension (:), intent (in out) :: g

!     integer :: n, iseg, llim, ulim
!     complex, dimension (:), allocatable :: tmp

!     n = size(g)

!     allocate (tmp(n))

!     call first_order_upwind (1,g,delzed(0),stream_sign(iv),tmp)

! !    do iseg = 1, size(tmp)
! !       write (*,*) 'gimp_2', ivmu, iseg, cabs(tmp(iseg)), cabs(g(iseg))
! !    end do

!     iseg = 1
!     llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
! !    g(llim:ulim) = g(llim:ulim) + 2.*stream(ig_low(iseg):ig_up(iseg),iv,is)*tmp(llim:ulim)
!     g(llim:ulim) = g(llim:ulim) + stream(ig_low(iseg):ig_up(iseg),iv,is)*tmp(llim:ulim)
!     if (nsegments(ie,iky) > 1) then
!        do iseg = 2, nsegments(ie,iky)
! !          llim = (iseg-1)*nz+2
! !          ulim = iseg*nz+1
!           llim = ulim+1
!           ulim = llim+ig_up(iseg)-ig_low(iseg)-1
!           g(llim:ulim) = g(llim:ulim) + stream(ig_low(iseg)+1:ig_up(iseg),iv,is)*tmp(llim:ulim)
! !          g(llim:ulim) = g(llim:ulim) + 2.*stream(ig_low(iseg)+1:ig_up(iseg),iv,is)*tmp(llim:ulim)
!        end do
!     end if

! !    write (*,*) 'GETDGDZ', n, ulim

!     deallocate (tmp)

!   end subroutine get_dgdz_implicit

  subroutine stream_tridiagonal_solve (iky, ie, iv, is, g)

    use finite_differences, only: tridag

    implicit none

    integer, intent (in) :: iky, ie, iv, is
    complex, dimension (:), intent (in out) :: g

    integer :: iseg, llim, ulim
    integer :: nz, nseg_max, sgn, n_ext
    real, dimension (:), allocatable :: a, b, c

    ! avoid double-counting at boundaries between 2pi segments
    nz = maxval(ig_up-ig_low)
    nseg_max = nsegments(ie,iky)
    sgn = stream_sign(iv)

    n_ext = nseg_max*nz+1
    allocate (a(n_ext))
    allocate (b(n_ext))
    allocate (c(n_ext))

    iseg = 1
    llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
    ! BACKWARDS DIFFERENCE FLAG
    a(llim:ulim) = -2.0*stream(ig_low(iseg):ig_up(iseg),iv,is)*stream_tri_a(llim:ulim,sgn)
    b(llim:ulim) = 1.0-2.0*stream(ig_low(iseg):ig_up(iseg),iv,is)*stream_tri_b(llim:ulim,sgn)
    c(llim:ulim) = -2.0*stream(ig_low(iseg):ig_up(iseg),iv,is)*stream_tri_c(llim:ulim,sgn)

    if (nsegments(ie,iky) > 1) then
       do iseg = 2, nsegments(ie,iky)
          llim = ulim+1
          ulim = llim+ig_up(iseg)-ig_low(iseg)-1
    ! BACKWARDS DIFFERENCE FLAG
          a(llim:ulim) = -2.0*stream(ig_low(iseg)+1:ig_up(iseg),iv,is)*stream_tri_a(llim:ulim,sgn)
          b(llim:ulim) = 1.0-2.0*stream(ig_low(iseg)+1:ig_up(iseg),iv,is)*stream_tri_b(llim:ulim,sgn)
          c(llim:ulim) = -2.0*stream(ig_low(iseg)+1:ig_up(iseg),iv,is)*stream_tri_c(llim:ulim,sgn)
       end do
    end if
    a(ulim) = -2.0*stream(ig_up(nsegments(ie,iky)),iv,is)*stream_tri_a(size(stream_tri_a,1),sgn)
    b(ulim) = 1.0-2.0*stream(ig_up(nsegments(ie,iky)),iv,is)*stream_tri_b(size(stream_tri_b,1),sgn)
    call tridag (1, a(:ulim), b(:ulim), c(:ulim), g)

    deallocate (a, b, c)

  end subroutine stream_tridiagonal_solve

  subroutine stream_phi_tridiagonal_solve (iky, ie, phi)

    use finite_differences, only: tridag

    implicit none

    integer, intent (in) :: iky, ie
    complex, dimension (:), intent (in out) :: phi

    integer :: iseg, ikx, llim, ulim
    integer :: nz, nseg_max, n_ext
    real, dimension (:), allocatable :: a, b, c

    ! avoid double-counting at boundaries between 2pi segments
    nz = maxval(ig_up-ig_low)
    nseg_max = nsegments(ie,iky)

    n_ext = nseg_max*nz+1
    allocate (a(n_ext))
    allocate (b(n_ext))
    allocate (c(n_ext))

    iseg = 1
    ikx = ikxmod(iseg,ie,iky)
    llim = 1 ; ulim = ig_up(iseg)-ig_low(iseg)+1
    ! BACKWARDS DIFFERENCE FLAG
    a(llim:ulim) = gam_stream(iky,ikx,ig_low(iseg):ig_up(iseg))*stream_tri_diff_a(llim:ulim)
    b(llim:ulim) = 1.0+gam_stream(iky,ikx,ig_low(iseg):ig_up(iseg))*stream_tri_diff_b(llim:ulim)
    c(llim:ulim) = gam_stream(iky,ikx,ig_low(iseg):ig_up(iseg))*stream_tri_diff_c(llim:ulim)

    if (nsegments(ie,iky) > 1) then
       do iseg = 2, nsegments(ie,iky)
          ikx = ikxmod(iseg,ie,iky)
          llim = ulim+1
          ulim = llim+ig_up(iseg)-ig_low(iseg)-1
    ! BACKWARDS DIFFERENCE FLAG
          a(llim:ulim) = gam_stream(iky,ikx,ig_low(iseg)+1:ig_up(iseg))*stream_tri_diff_a(llim:ulim)
          b(llim:ulim) = 1.0+gam_stream(iky,ikx,ig_low(iseg)+1:ig_up(iseg))*stream_tri_diff_b(llim:ulim)
          c(llim:ulim) = gam_stream(iky,ikx,ig_low(iseg)+1:ig_up(iseg))*stream_tri_diff_c(llim:ulim)
       end do
    end if
    ikx = ikxmod(nsegments(ie,iky),ie,iky)
    a(ulim) = gam_stream(iky,ikx,ig_up(nsegments(ie,iky)))*stream_tri_diff_a(size(stream_tri_diff_a))
    b(ulim) = 1.0+gam_stream(iky,ikx,ig_up(nsegments(ie,iky)))*stream_tri_diff_b(size(stream_tri_diff_b))
    call tridag (1, a(:ulim), b(:ulim), c(:ulim), phi)

    deallocate (a, b, c)

  end subroutine stream_phi_tridiagonal_solve

  subroutine advance_wdrifty_implicit (g)

    use constants, only: zi
    use stella_layouts, only: vmu_lo
    use zgrid, only: nzgrid
    use kt_grids, only: aky, naky
    use dist_fn_arrays, only: wdrifty

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g

    integer :: ivmu, iky, iz
    complex :: tmp

    do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
       do iz = -nzgrid, nzgrid
          do iky = 1, naky
             tmp = 0.5*zi*aky(iky)*wdrifty(1,iz,ivmu)
             g(iky,:,iz,ivmu) = g(iky,:,iz,ivmu) * (1.0 + tmp) / (1.0 - tmp)
!             g(iky,:,iz,ivmu) = g(iky,:,iz,ivmu) * (1.0 + 2.0*tmp)
          end do
       end do
    end do

  end subroutine advance_wdrifty_implicit

  subroutine advance_wstar_implicit (g, phi, apar)

    use constants, only: zi
    use stella_layouts, only: vmu_lo
    use stella_layouts, only: iv_idx, is_idx
    use run_parameters, only: fphi, fapar
    use zgrid, only: nzgrid
    use dist_fn_arrays, only: gstar_to_g

    implicit none

    complex, dimension (:,:,-nzgrid:,vmu_lo%llim_proc:), intent (in out) :: g
    complex, dimension (:,:,-nzgrid:), intent (in out) :: phi, apar

    ! given g^{*}, obtain phi^{*} and apar^{*}
!    call advance_fields (g, phi, apar, dist='gbar')

    ! solve g^{**} = g^{*}+i*wstar*ky*J0*(chi^{**}+chi^{*})/2
    ! define gstar^{**} = g^{**} - i*wstar*ky*J0*chi^{**}/2
    ! so that gstar^{**} = g^{*} + i*wstar*ky*J0*chi^{*}/2

    ! BACKWARDS DIFFERENCE FLAG
!     do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
!        iv = iv_idx(vmu_lo,ivmu)
!        is = is_idx(vmu_lo,ivmu)
!        do iz = -nzgrid, nzgrid
!           do ikx = 1, nakx
!              do iky = 1, naky
!                 g(iky,ikx,iz,ivmu) = g(iky,ikx,iz,ivmu) &
!                      + zi*aky(iky)*wstar(iz,ivmu)*aj0x(iky,ikx,iz,ivmu) &
!                      * (fphi*phi(iky,ikx,iz) - fapar*vpa(iv)*spec(is)%stm*apar(iky,ikx,iz))
!              end do
!           end do
!        end do
!     end do

    ! now that we have gstar^{**}=g^{*}, obtain corresponding fields, phi^{**} and apar^{**}
    call advance_fields (g, phi, apar, dist='gstar')
    ! convert from gstar^{**} to g^{**}
    call gstar_to_g (g, phi, apar, fphi, fapar)

  end subroutine advance_wstar_implicit

  subroutine finish_dist_fn

    use stella_transforms, only: finish_transforms
    use kt_grids, only: alpha_global
    
    implicit none

    dist_fn_initialized = .false.
    readinit = .false.
    gxyz_initialized = .false.

    if (alpha_global) call finish_transforms
    call finish_redistribute
    call finish_stream
    call finish_mirror
    call finish_bessel
    call finish_wdrift
    call finish_wstar
    call finish_kperp2
    call deallocate_arrays

  end subroutine finish_dist_fn

  subroutine finish_redistribute

    implicit none

    redistinit = .false.

  end subroutine finish_redistribute

  subroutine finish_stream

    implicit none

    if (allocated(stream)) deallocate (stream)
    if (allocated(stream_sign)) deallocate (stream_sign)

    if (stream_implicit) call finish_invert_stream_operator

    streaminit = .false.

  end subroutine finish_stream

  subroutine finish_invert_stream_operator

    implicit none

    if (allocated(stream_tri_a)) then
       deallocate (stream_tri_a)
       deallocate (stream_tri_b)
       deallocate (stream_tri_c)
    end if

  end subroutine finish_invert_stream_operator

  subroutine finish_mirror

!    use sherman_morrison, only: finish_invert_mirror_operator

    implicit none

    if (allocated(mirror)) deallocate (mirror)
    if (allocated(mirror_sign)) deallocate (mirror_sign)

    if (mirror_implicit) call finish_invert_mirror_operator

    mirrorinit = .false.

  end subroutine finish_mirror

  subroutine finish_invert_mirror_operator

    implicit none

    if (allocated(mirror_tri_a)) then
       deallocate (mirror_tri_a)
       deallocate (mirror_tri_b)
       deallocate (mirror_tri_c)
    end if

  end subroutine finish_invert_mirror_operator

  subroutine finish_wdrift

    use dist_fn_arrays, only: wdriftx, wdrifty

    implicit none

    if (allocated(wdriftx)) deallocate (wdriftx)
    if (allocated(wdrifty)) deallocate (wdrifty)

    wdriftinit = .false.

  end subroutine finish_wdrift

  subroutine finish_wstar

    use dist_fn_arrays, only: wstar

    implicit none

    if (allocated(wstar)) deallocate (wstar)

    wstarinit = .false.

  end subroutine finish_wstar

  subroutine finish_kperp2

    use dist_fn_arrays, only: kperp2

    implicit none

    if (allocated(kperp2)) deallocate (kperp2)

    kp2init = .false.

  end subroutine finish_kperp2

  subroutine deallocate_arrays

    use dist_fn_arrays, only: gnew, gold
    use dist_fn_arrays, only: g1, g2, g3

    implicit none

    if (allocated(gnew)) deallocate (gnew)
    if (allocated(gold)) deallocate (gold)
    if (allocated(g1)) deallocate (g1)
    if (allocated(g2)) deallocate (g2)
    if (allocated(g3)) deallocate (g3)

  end subroutine deallocate_arrays

  subroutine finish_bessel

    use dist_fn_arrays, only: aj0v, aj0x

    implicit none

    if (allocated(aj0v)) deallocate (aj0v)
    if (allocated(aj0x)) deallocate (aj0x)

    bessinit = .false.

  end subroutine finish_bessel

end module dist_fn
