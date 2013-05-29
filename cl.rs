#[link(name = "cl", vers = "0.1", uuid = "1205b5b0-fbd0-4eeb-bfac-e85947419b4e")];

#[license = "MIT"];
#[crate_type = "lib"];

#[author = "Jens Nockert"];

#[comment = "OpenCL library for Rust"];
#[desc = "Trying to make OpenCL more usable from Rust, uses the openlc.rs bindings."];

extern mod vector (uuid = "bd080482-e281-48ee-bb19-917f6b4a466f");
extern mod string (uuid = "86237a28-e038-4119-a0be-df3fda9521de");
extern mod opencl (uuid = "f83bfc2b-e3ee-4e4c-b324-70e379fbcff2");

use vector::Iterators;
use string::Split;
use opencl::*;

use core::result;

macro_rules! cl_call(($name:ident, $error:ident: $($arg:expr),+) => ({
    let $error = unsafe { $name($($arg),+) };

    if ($error != CL_SUCCESS) {
        return result::Err($error);
    }
}))


macro_rules! cl_call_unknown_length(
($name:ident, $n:ty, $in:ty, $out:ty: $($arg:expr),+) => ({
    let mut n:$n = 0;

    cl_call!($name, err1: $($arg),+, 0, ptr::mut_null(), ptr::to_mut_unsafe_ptr(&mut n));

    let mut result:~[$out] = vec::with_capacity(n as uint);

    cl_call!($name, err2: $($arg),+, n, vec::raw::to_mut_ptr(result) as *mut $in, ptr::mut_null());

    unsafe { vec::raw::set_len(&mut result, n as uint) };

    result
});
($name:ident, $n:ty, $in:ty, $out:ty) => ({
    let mut n:$n = 0;

    cl_call!($name, err1: 0, ptr::mut_null(), ptr::to_mut_unsafe_ptr(&mut n));

    let mut result:~[$out] = vec::with_capacity(n as uint);

    cl_call!($name, err2: n, vec::raw::to_mut_ptr(result) as *mut $in, ptr::mut_null());

    unsafe { vec::raw::set_len(&mut result, n as uint) };

    result
}))

macro_rules! cl_get_info(($name:expr) => (match self.get_info($name) {
    Ok(x) => x,
    Err(n) => fail!(fmt!("Error %? should not happen for %?", n, $name))
}))

macro_rules! cl_extract_int(($t:ty, $v:expr) => (vec::foldr($v, 0 as $t, |n, sum| { (256 * sum) + (*n as $t) })))
macro_rules! cl_extract_ints(($t:ty, $v:expr) => (vec::map($v.to_cons(sys::size_of::<$t>()), |v| { cl_extract_int!($t, *v) })))
macro_rules! cl_extract_string(($v:expr) => (unsafe {
    let value = $v;
    str::raw::from_bytes_with_null(value).to_owned()
}))

struct Platform {
    id: cl_platform_id
}

struct Device {
    id: cl_device_id
}

impl Clone for Platform {
    fn clone(&self) -> Platform {
        Platform { id: self.id }
    }
}

impl Clone for Device {
    fn clone(&self) -> Device {
        Device { id: self.id }
    }
}

impl Platform {
    pub fn all() -> result::Result<~[Platform], cl_int> {
        return result::Ok(cl_call_unknown_length!(clGetPlatformIDs, cl_uint, cl_platform_id, Platform));
    }

    pub fn get_info(&self, name: cl_platform_info) -> result::Result<~str, cl_int> {
        return result::Ok(cl_extract_string!(cl_call_unknown_length!(clGetPlatformInfo, libc::size_t, libc::c_void, u8: self.id, name)));
    }

    pub fn profile(&self) -> ~str { cl_get_info!(CL_PLATFORM_PROFILE) }
    pub fn version(&self) -> ~str { cl_get_info!(CL_PLATFORM_VERSION) }
    pub fn name(&self) -> ~str { cl_get_info!(CL_PLATFORM_NAME) }
    pub fn vendor(&self) -> ~str { cl_get_info!(CL_PLATFORM_VENDOR) }
    pub fn extensions(&self) -> ~[~str] { let result = cl_get_info!(CL_PLATFORM_EXTENSIONS); result.split(' ') }

    pub fn devices(&self, device_type: cl_device_type) -> result::Result<~[Device], cl_int> {
        return result::Ok(cl_call_unknown_length!(clGetDeviceIDs, cl_uint, cl_device_id, Device: self.id, device_type));
    }
    
    pub fn unload_compiler(&self) -> result::Result<bool, cl_int> {
        cl_call!(clUnloadPlatformCompiler, err1: self.id);
        
        return result::Ok(true)
    }
}

impl Device {
    pub fn get_info(&self, name: cl_device_info) -> result::Result<~[u8], cl_int> {
        return result::Ok(cl_call_unknown_length!(clGetDeviceInfo, libc::size_t, libc::c_void, u8: self.id, name));
    }

    pub fn device_type(&self) -> cl_device_type { cl_extract_int!(cl_device_type, cl_get_info!(CL_DEVICE_TYPE)) } // TODO: Why doesn't rust like `type' as a name?
    pub fn vendor_id(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_VENDOR_ID)) }
    pub fn max_compute_units(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_COMPUTE_UNITS)) }
    pub fn max_work_item_dimensions(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)) }
    pub fn max_work_group_size(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_MAX_WORK_GROUP_SIZE)) }
    pub fn max_work_item_sizes(&self) -> ~[libc::size_t] { cl_extract_ints!(libc::size_t, cl_get_info!(CL_DEVICE_MAX_WORK_ITEM_SIZES)) }
    pub fn preferred_vector_width_char(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)) }
    pub fn preferred_vector_width_short(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT)) }
    pub fn preferred_vector_width_int(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT)) }
    pub fn preferred_vector_width_long(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG)) }
    pub fn preferred_vector_width_float(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)) }
    pub fn preferred_vector_width_double(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)) }
    pub fn max_clock_frequency(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_CLOCK_FREQUENCY)) }
    pub fn address_bits(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_ADDRESS_BITS)) }
    pub fn max_read_image_args(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_READ_IMAGE_ARGS)) }
    pub fn max_write_image_args(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_WRITE_IMAGE_ARGS)) }
    pub fn max_mem_alloc_size(&self) -> cl_ulong { cl_extract_int!(cl_ulong, cl_get_info!(CL_DEVICE_MAX_MEM_ALLOC_SIZE)) }
    pub fn image2d_max_width(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE2D_MAX_WIDTH)) }
    pub fn image2d_max_height(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE2D_MAX_HEIGHT)) }
    pub fn image3d_max_width(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE3D_MAX_WIDTH)) }
    pub fn image3d_max_height(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE3D_MAX_HEIGHT)) }
    pub fn image3d_max_depth(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE3D_MAX_DEPTH)) }
    pub fn image_support(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_IMAGE_SUPPORT))) }
    pub fn max_parameter_size(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_MAX_PARAMETER_SIZE)) }
    pub fn max_samplers(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_SAMPLERS)) }
    pub fn mem_base_addr_align(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MEM_BASE_ADDR_ALIGN)) }
    pub fn min_data_type_align_size(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)) }
    pub fn single_fp_config(&self) -> cl_device_fp_config { cl_extract_int!(cl_device_fp_config, cl_get_info!(CL_DEVICE_SINGLE_FP_CONFIG)) }
    pub fn global_mem_cache_type(&self) -> cl_device_mem_cache_type { cl_extract_int!(cl_device_mem_cache_type, cl_get_info!(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE)) }
    pub fn global_mem_cacheline_size(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)) }
    pub fn global_mem_cache_size(&self) -> cl_ulong { cl_extract_int!(cl_ulong, cl_get_info!(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)) }
    pub fn global_mem_size(&self) -> cl_ulong { cl_extract_int!(cl_ulong, cl_get_info!(CL_DEVICE_GLOBAL_MEM_SIZE)) }
    pub fn max_constant_buffer_size(&self) -> cl_ulong { cl_extract_int!(cl_ulong, cl_get_info!(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)) }
    pub fn max_constant_args(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_MAX_CONSTANT_ARGS)) }
    pub fn local_mem_type(&self) -> cl_device_local_mem_type { cl_extract_int!(cl_device_local_mem_type, cl_get_info!(CL_DEVICE_LOCAL_MEM_TYPE)) }
    pub fn local_mem_size(&self) -> cl_ulong { cl_extract_int!(cl_ulong, cl_get_info!(CL_DEVICE_LOCAL_MEM_SIZE)) }
    pub fn error_correction_support(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_ERROR_CORRECTION_SUPPORT))) }
    pub fn profiling_timer_resolution(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_PROFILING_TIMER_RESOLUTION)) }
    pub fn endian_little(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_ENDIAN_LITTLE))) }
    pub fn available(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_AVAILABLE))) }
    pub fn compiler_available(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_COMPILER_AVAILABLE))) }
    pub fn execution_capabilities(&self) -> cl_device_exec_capabilities { cl_extract_int!(cl_device_exec_capabilities, cl_get_info!(CL_DEVICE_EXECUTION_CAPABILITIES)) }
    pub fn queue_properties(&self) -> cl_command_queue_properties { cl_extract_int!(cl_command_queue_properties, cl_get_info!(CL_DEVICE_QUEUE_PROPERTIES)) }
    pub fn name(&self) -> ~str { cl_extract_string!(cl_get_info!(CL_DEVICE_NAME)) }
    pub fn vendor(&self) -> ~str { cl_extract_string!(cl_get_info!(CL_DEVICE_VENDOR)) }
    pub fn driver_version(&self) -> ~str { cl_extract_string!(cl_get_info!(CL_DRIVER_VERSION)) }
    pub fn profile(&self) -> ~str { cl_extract_string!(cl_get_info!(CL_DEVICE_PROFILE)) }
    pub fn version(&self) -> ~str { cl_extract_string!(cl_get_info!(CL_DEVICE_VERSION)) }
    pub fn extensions(&self) -> ~[~str] { let result = cl_extract_string!(cl_get_info!(CL_DEVICE_EXTENSIONS)); result.split(' ') }
    pub fn platform(&self) -> Platform { Platform { id: cl_extract_int!(uint, cl_get_info!(CL_DEVICE_PLATFORM)) as cl_platform_id } }

    pub fn preferred_vector_width_half(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF)) }
    pub fn host_unified_memory(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_HOST_UNIFIED_MEMORY))) }
    pub fn native_vector_width_char(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR)) }
    pub fn native_vector_width_short(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT)) }
    pub fn native_vector_width_int(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT)) }
    pub fn native_vector_width_long(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG)) }
    pub fn native_vector_width_float(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT)) }
    pub fn native_vector_width_double(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE)) }
    pub fn native_vector_width_half(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF)) }
    pub fn opencl_c_version(&self) -> ~str { cl_extract_string!(cl_get_info!(CL_DEVICE_OPENCL_C_VERSION)) }

    pub fn double_fp_config(&self) -> cl_device_fp_config { cl_extract_int!(cl_device_fp_config, cl_get_info!(CL_DEVICE_DOUBLE_FP_CONFIG)) }
    pub fn linker_available(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_LINKER_AVAILABLE))) }
    pub fn built_in_kernels(&self) -> ~[~str] { let result = cl_extract_string!(cl_get_info!(CL_DEVICE_BUILT_IN_KERNELS)); result.split(';') }
    pub fn image_max_buffer_size(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)) }
    pub fn image_max_array_size(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE)) }
    pub fn parent_device(&self) -> Device { Device { id: cl_extract_int!(uint, cl_get_info!(CL_DEVICE_PARENT_DEVICE)) as cl_device_id } }
    pub fn partition_max_sub_devices(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_PARTITION_MAX_SUB_DEVICES)) }
    pub fn partition_properties(&self) -> ~[cl_device_partition_property] { cl_extract_ints!(cl_device_partition_property, cl_get_info!(CL_DEVICE_MAX_WORK_ITEM_SIZES)) }
    pub fn partition_affinity_domain(&self) -> cl_device_affinity_domain { cl_extract_int!(cl_device_affinity_domain, cl_get_info!(CL_DEVICE_PARTITION_MAX_SUB_DEVICES)) }
    pub fn partition_type(&self) -> ~[cl_device_partition_property] { cl_extract_ints!(cl_device_partition_property, cl_get_info!(CL_DEVICE_PARTITION_TYPE)) }
    pub fn reference_count(&self) -> cl_uint { cl_extract_int!(cl_uint, cl_get_info!(CL_DEVICE_REFERENCE_COUNT)) }
    pub fn preferred_interop_user_sync(&self) -> bool { CL_TRUE == (cl_extract_int!(cl_bool, cl_get_info!(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC))) }
    pub fn printf_buffer_size(&self) -> libc::size_t { cl_extract_int!(libc::size_t, cl_get_info!(CL_DEVICE_PRINTF_BUFFER_SIZE)) }

    // cl_constant!(cl_device_info, CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A)        // TODO: Not available in online man-pages
    // cl_constant!(cl_device_info, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B) // TODO: Not available in online man-pages

}
