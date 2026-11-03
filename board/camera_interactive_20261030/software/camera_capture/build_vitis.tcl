# XSCT 2024.2: isolated standalone A53 camera firmware, FSBL and PMU firmware.
# Usage: xsct build_vitis.tcl [all|bsp] [path/to/camera_capture.xsa]
# Source inputs are restricted to this directory's C/header files. Existing OMR
# and score-display projects are never imported or modified.
set source_dir [file normalize [file dirname [info script]]]
set repository_dir [file normalize [file join $source_dir .. ..]]
set workspace_dir [file join $repository_dir build camera_capture_vitis]
set artifact_dir [file join $workspace_dir artifacts]
set platform_name camera_capture_platform
set domain_name camera_standalone
set app_name camera_capture
set stage [expr {[llength $argv] ? [lindex $argv 0] : "all"}]
set xsa [expr {[llength $argv] > 1 ? [file normalize [lindex $argv 1]] :
    [file join $repository_dir build vivado_camera camera_capture.xsa]}]

proc require_file {path} {
    if {![file isfile $path] || [file size $path] == 0} {
        error "Required nonempty file is missing: $path"
    }
}

proc file_bytes {path} {
    set handle [open $path rb]
    set data [read $handle]
    close $handle
    return $data
}

proc copy_boot_artifact {platform_directory basename destination} {
    # Use only the BSP-built binaries. A platform also contains a prebuilt QEMU
    # pmufw.elf, which must never be mistaken for the hardware's boot firmware.
    if {$basename eq "fsbl.elf"} {
        set input [file join $platform_directory zynqmp_fsbl fsbl_a53.elf]
    } elseif {$basename eq "pmufw.elf"} {
        set input [file join $platform_directory zynqmp_pmufw pmufw.elf]
    } else {
        error "Unexpected boot artifact: $basename"
    }
    require_file $input
    file copy -force $input $destination
    puts "BOOT_ARTIFACT: $destination"
}

proc run_command {command} {
    # Preserve argv boundaries (including workspaces containing spaces). No
    # shell and no Vitis Classic/Eclipse application-creation service is needed.
    if {[catch {exec {*}$command 2>@1} output options]} {
        puts stderr $output
        return -options $options $output
    }
    if {$output ne ""} {puts $output}
}

proc build_application {source_directory app_directory bsp_directory} {
    global env
    if {[info exists env(VITIS_ROOT)]} {
        set gcc [file join $env(VITIS_ROOT) gnu aarch64 nt aarch64-none bin aarch64-none-elf-gcc.exe]
    } else {
        set gcc [lindex [auto_execok aarch64-none-elf-gcc] 0]
    }
    require_file $gcc
    set sources [file join $app_directory src]
    set output [file join $app_directory Release]
    file mkdir $sources
    file mkdir $output
    set objects {}
    foreach source [concat [glob -nocomplain -directory $source_directory *.c] \
                           [glob -nocomplain -directory $source_directory *.h]] {
        file copy -force $source $sources
    }
    require_file [file join $source_directory lscript.ld]
    file copy -force [file join $source_directory lscript.ld] [file join $sources lscript.ld]
    foreach source [lsort [glob -directory $source_directory *.c]] {
        set copied_source [file join $sources [file tail $source]]
        set object [file join $output [file rootname [file tail $source]].o]
        puts "COMPILE: [file tail $source]"
        run_command [list $gcc -mcpu=cortex-a53 -std=c11 -O2 -g3 -Wall -Wextra -Werror \
            -I[file join $bsp_directory include] -I$sources -c $copied_source -o $object]
        lappend objects $object
    }
    set elf [file join $output camera_capture.elf]
    puts "LINK: $elf"
    run_command [concat [list $gcc -mcpu=cortex-a53 -o $elf] $objects [list \
        -Wl,-T,[file join $sources lscript.ld] \
        -Wl,-Map,[file join $output camera_capture.map] \
        -L[file join $bsp_directory lib] \
        -Wl,--start-group -lxilffs -lxil -lgcc -lc -lm -Wl,--end-group]]
    require_file $elf
    return $elf
}

proc build_capture {} {
    global source_dir repository_dir workspace_dir artifact_dir platform_name
    global domain_name app_name stage xsa
    if {$stage ni {all bsp}} {
        error "Stage must be all or bsp, got: $stage"
    }
    require_file $xsa
    file mkdir $workspace_dir
    file mkdir $artifact_dir
    # A completion marker is written only after all requested outputs exist.
    file delete -force [file join $artifact_dir ${stage}.complete]
    setws $workspace_dir
    set platform_dir [file join $workspace_dir $platform_name]
    set platform_project [file join $platform_dir platform.spr]
    set imported_xsa [file join $platform_dir hw [file tail $xsa]]
    # XSCT 2024.2's in-place -updatehw may lose psu_init sources on Windows.
    # Retain the old generated platform and create a fresh one for a new XSA.
    # Both locations are fixed children of this task's generated workspace.
    if {[file exists $platform_project] &&
        (![file isfile $imported_xsa] || [file_bytes $imported_xsa] ne [file_bytes $xsa])} {
        set archive_dir [file join $workspace_dir ${platform_name}_previous_[clock seconds]]
        file rename $platform_dir $archive_dir
        puts "Previous generated platform retained at: $archive_dir"
    }
    set configure_bsp 1
    if {[file exists $platform_project]} {
        platform read $platform_project
        platform active $platform_name
        set bsp_mss [file join $platform_dir psu_cortexa53_0 $domain_name bsp system.mss]
        if {[file isfile $imported_xsa] &&
            [file_bytes $imported_xsa] eq [file_bytes $xsa] &&
            [file isfile $bsp_mss] &&
            [string first "LIBRARY_NAME = xilffs" [file_bytes $bsp_mss]] >= 0 &&
            [regexp {PARAMETER use_lfn = 1\y} [file_bytes $bsp_mss]]} {
            set configure_bsp 0
        }
    } else {
        platform create -name $platform_name -hw $xsa
        platform active $platform_name
        domain create -name $domain_name -proc psu_cortexa53_0 -os standalone -arch 64-bit
    }
    domain active $domain_name
    if {$configure_bsp} {
        bsp setlib -name xilffs
        bsp config stdin psu_uart_1
        bsp config stdout psu_uart_1
        bsp config fs_interface 1
        # Preserve the requested mixed-case directory name Images. Mode 1 uses
        # FatFs's static working buffer; this application is single-threaded.
        bsp config use_lfn 1
        bsp write
        platform write
    }
    set bsp_libdir [file join $platform_dir psu_cortexa53_0 $domain_name bsp psu_cortexa53_0 lib]
    set exported_platform [file join $platform_dir export $platform_name ${platform_name}.xpfm]
    if {$configure_bsp || ![file isfile [file join $bsp_libdir libxil.a]] ||
        ![file isfile [file join $bsp_libdir libxilffs.a]] ||
        ![file isfile [file join $platform_dir zynqmp_fsbl fsbl_a53.elf]] ||
        ![file isfile [file join $platform_dir zynqmp_pmufw pmufw.elf]] ||
        ![file isfile $exported_platform]} {
        platform generate
    } else {
        puts "Reusing the generated BSP and boot firmware for the identical XSA."
    }
    copy_boot_artifact $platform_dir fsbl.elf [file join $artifact_dir fsbl.elf]
    copy_boot_artifact $platform_dir pmufw.elf [file join $artifact_dir pmufw.elf]

    if {$stage eq "all"} {
        require_file [file join $source_dir main.c]
        set app_dir [file join $workspace_dir $app_name]
        set application_elf [build_application $source_dir $app_dir [file dirname $bsp_libdir]]
        file copy -force $application_elf [file join $artifact_dir ${app_name}.elf]
        file copy -force [file join $app_dir Release ${app_name}.map] \
            [file join $artifact_dir ${app_name}.map]
        puts "APP_ARTIFACT: [file join $artifact_dir ${app_name}.elf]"
        set bitstream [file join $repository_dir build vivado_camera camera_capture.runs impl_1 camera_capture_wrapper.bit]
        set bif [open [file join $artifact_dir camera_capture.bif] w]
        puts $bif "the_ROM_image:"
        puts $bif "\{"
        puts $bif "  \[bootloader, destination_cpu=a53-0\] \"[file join $artifact_dir fsbl.elf]\""
        puts $bif "  \[pmufw_image\] \"[file join $artifact_dir pmufw.elf]\""
        puts $bif "  \[destination_device=pl\] \"$bitstream\""
        puts $bif "  \[destination_cpu=a53-0, exception_level=el-3, trustzone=secure\] \"[file join $artifact_dir ${app_name}.elf]\""
        puts $bif "\}"
        close $bif
    }
    set marker [open [file join $artifact_dir ${stage}.complete] w]
    puts $marker "stage=$stage"
    puts $marker "xsa=$xsa"
    puts $marker "completed=[clock format [clock seconds] -format {%Y-%m-%dT%H:%M:%S%z}]"
    close $marker
    puts "CAMERA_BUILD_COMPLETE: $stage"
}

if {[catch {build_capture} message options]} {
    puts stderr "CAMERA_BUILD_FAILED: $message"
    if {[dict exists $options -errorinfo]} {
        puts stderr [dict get $options -errorinfo]
    }
    exit 1
}
exit 0
