#ifndef TEST_FF_H
#define TEST_FF_H
#include <stdint.h>
typedef unsigned UINT;
typedef uint32_t FSIZE_t;
typedef int FRESULT;
enum {FR_OK,FR_NO_FILE,FR_NO_PATH,FR_EXIST,FR_DISK_ERR};
enum {FA_READ=1,FA_WRITE=2,FA_CREATE_NEW=4,AM_DIR=16};
typedef struct { unsigned csize; } FATFS;
typedef struct { FSIZE_t position,size; } FIL;
typedef struct { int dummy; } DIR;
typedef struct { char fname[128];unsigned fattrib; } FILINFO;
#define f_size(f) ((f)->size)
#define f_tell(f) ((f)->position)
FRESULT f_mount(FATFS *,const char *,unsigned);
FRESULT f_mkdir(const char *);
FRESULT f_opendir(DIR *,const char *);
FRESULT f_readdir(DIR *,FILINFO *);
FRESULT f_closedir(DIR *);
FRESULT f_open(FIL *,const char *,unsigned);
FRESULT f_read(FIL *,void *,UINT,UINT *);
FRESULT f_write(FIL *,const void *,UINT,UINT *);
FRESULT f_sync(FIL *);
FRESULT f_close(FIL *);
FRESULT f_lseek(FIL *,FSIZE_t);
FRESULT f_rename(const char *,const char *);
FRESULT f_unlink(const char *);
#endif
