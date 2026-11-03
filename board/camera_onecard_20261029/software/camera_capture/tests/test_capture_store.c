#include "capture_store.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char path[CAPTURE_PATH_CAPACITY];
    uint8_t *bytes;
    size_t size;
    size_t capacity;
    int exists;
    int is_directory;
} memory_file;

typedef struct {
    memory_file files[24];
    int offline, writes, fail_write, short_write, fail_sync, fail_close, fail_rename;
    int fail_row;
    int reads;
    int multiple_directories, rename_calls, fail_rename_at, fail_remove_final;
    int sync_calls, close_calls, fail_sync_at, fail_close_at;
    const char *directory;
} memory_fs;

static int same_path(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        ++a;
        ++b;
    }
    return *a == *b;
}

static const char *directory_name(const memory_fs *fs)
{
    return fs->directory ? fs->directory : "0:/Image";
}

static memory_file *find(memory_fs *fs, const char *path)
{
    unsigned i;
    for (i = 0; i < 24; ++i)
        if (fs->files[i].exists && same_path(fs->files[i].path, path))
            return &fs->files[i];
    return NULL;
}

static int make_dir(void *context, const char *path)
{
    memory_fs *fs = context;
    assert(fs->multiple_directories || strcmp(path, directory_name(fs)) == 0);
    return fs->offline ? CAPTURE_FS_IO : CAPTURE_OK;
}

static int list(void *context, const char *path, capture_list_visitor visitor, void *vctx)
{
    memory_fs *fs = context;
    unsigned i;
    if (fs->offline) return CAPTURE_FS_IO;
    for (i = 0; i < 24; ++i) {
        memory_file *f = &fs->files[i];
        if (f->exists) {
            if(strncmp(path,f->path,strlen(path))!=0 || f->path[strlen(path)]!='/') continue;
            if (visitor(vctx, f->path + strlen(path) + 1, f->is_directory) != CAPTURE_OK)
                return CAPTURE_FS_IO;
        }
    }
    return CAPTURE_OK;
}

static int read_small(void *context, const char *path, uint8_t *data, size_t cap, size_t *length)
{
    memory_fs *fs = context;
    memory_file *f;
    ++fs->reads;
    if (fs->offline) return CAPTURE_FS_IO;
    f = find(fs, path);
    if (!f) return CAPTURE_FS_NOENT;
    if (f->size > cap) return CAPTURE_STATE_CORRUPT;
    *length = f->size;
    if (f->size) memcpy(data, f->bytes, f->size);
    return CAPTURE_OK;
}

static int create(void *context, const char *path, void **handle)
{
    memory_fs *fs = context;
    unsigned i;
    if (fs->offline) return CAPTURE_FS_IO;
    if (find(fs, path)) return CAPTURE_FS_EXISTS;
    for (i = 0; i < 24; ++i) {
        memory_file *f = &fs->files[i];
        if (!f->exists) {
            free(f->bytes);
            memset(f, 0, sizeof(*f));
            strcpy(f->path, path);
            f->exists = 1;
            *handle = f;
            return CAPTURE_OK;
        }
    }
    return CAPTURE_FS_NO_SPACE;
}

static int write_bytes(void *context, void *handle, const uint8_t *data,
                        size_t length, size_t *written)
{
    memory_fs *fs = context;
    memory_file *f = handle;
    size_t amount = length;
    ++fs->writes;
    *written = 0;
    if (fs->offline || fs->writes == fs->fail_write) return CAPTURE_FS_IO;
    if (fs->writes == fs->short_write) --amount;
    if (f->size + amount > f->capacity) {
        size_t capacity = f->capacity ? f->capacity : 64u;
        while (capacity < f->size + amount) capacity *= 2u;
        f->bytes = realloc(f->bytes, capacity);
        assert(f->bytes);
        f->capacity = capacity;
    }
    memcpy(f->bytes + f->size, data, amount);
    f->size += amount;
    *written = amount;
    return CAPTURE_OK;
}

static int sync_file(void *context, void *handle)
{
    memory_fs *fs = context;
    (void)handle;
    ++fs->sync_calls;
    return fs->offline || fs->fail_sync || fs->sync_calls==fs->fail_sync_at ? CAPTURE_FS_IO : CAPTURE_OK;
}

static int close_file(void *context, void *handle)
{
    memory_fs *fs = context;
    (void)handle;
    ++fs->close_calls;
    return fs->offline || fs->fail_close || fs->close_calls==fs->fail_close_at ? CAPTURE_FS_IO : CAPTURE_OK;
}

static int rename_file(void *context, const char *from, const char *to)
{
    memory_fs *fs = context;
    memory_file *f;
    ++fs->rename_calls;
    if (fs->offline || fs->fail_rename || fs->rename_calls==fs->fail_rename_at) return CAPTURE_FS_IO;
    if (find(fs, to)) return CAPTURE_FS_EXISTS;
    f = find(fs, from);
    if (!f) return CAPTURE_FS_NOENT;
    strcpy(f->path, to);
    return CAPTURE_OK;
}

static int remove_file(void *context, const char *path)
{
    memory_fs *fs = context;
    memory_file *f;
    if (fs->offline) return CAPTURE_FS_IO;
    if(fs->fail_remove_final && strstr(path,".bmp")) return CAPTURE_FS_IO;
    f = find(fs, path);
    if (!f) return CAPTURE_FS_NOENT;
    f->exists = 0;
    free(f->bytes);
    f->bytes = NULL;
    f->size = 0;
    f->capacity = 0;
    return CAPTURE_OK;
}

static const capture_fs_ops operations = {
    make_dir, list, read_small, create, write_bytes, sync_file, close_file,
    rename_file, remove_file, NULL
};

static int rgb_row(void *context, uint32_t y, uint8_t *rgb, size_t length)
{
    memory_fs *fs = context;
    static const uint8_t rows[2][6] = {{255, 0, 0, 0, 255, 0}, {0, 0, 255, 255, 255, 255}};
    assert(y < 2 && length == 6);
    if (fs->fail_row) return CAPTURE_FS_IO;
    memcpy(rgb, rows[y], 6);
    return CAPTURE_OK;
}

typedef struct { uint32_t last_y; unsigned calls; } full_frame;

static int full_rgb_row(void *context, uint32_t y, uint8_t *rgb, size_t length)
{
    full_frame *frame = context;
    size_t x;
    assert(length == 4056u * 3u && y < 3040u);
    if (frame->calls) assert(y + 1u == frame->last_y);
    frame->last_y = y;
    ++frame->calls;
    for (x = 0; x < length; x += 3u) {
        rgb[x] = (uint8_t)y;
        rgb[x + 1u] = (uint8_t)(x / 3u);
        rgb[x + 2u] = 127;
    }
    return CAPTURE_OK;
}

static void reset(memory_fs *fs)
{
    unsigned i;
    for (i = 0; i < 24; ++i) free(fs->files[i].bytes);
    memset(fs, 0, sizeof(*fs));
}

static void add_file(memory_fs *fs, const char *name)
{
    char path[CAPTURE_PATH_CAPACITY];
    void *handle;
    snprintf(path, sizeof(path), "%s/%s", directory_name(fs), name);
    assert(create(fs, path, &handle) == CAPTURE_OK);
}

static void add_directory(memory_fs *fs, const char *name)
{
    char path[CAPTURE_PATH_CAPACITY];
    add_file(fs, name);
    snprintf(path, sizeof(path), "%s/%s", directory_name(fs), name);
    find(fs, path)->is_directory = 1;
}

static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int save(capture_store *s, memory_fs *fs, char name[CAPTURE_NAME_CAPACITY])
{
    uint8_t row[8];
    return capture_store_save_bmp(s, 2, 2, rgb_row, fs, row, sizeof(row), name);
}

static void init_three(capture_store s[3],memory_fs *fs)
{
    const char *paths[3]={"0:/Image","0:/Grey","0:/Binary"};
    unsigned i;
    fs->multiple_directories=1;
    for(i=0;i<3u;++i)
        assert(capture_store_init_sequence(&s[i],&operations,fs,paths[i])==CAPTURE_OK);
    assert(capture_store_align_sequences(s,3)==CAPTURE_OK);
}

static void test_pair(void)
{
    memory_fs fs={0}; capture_store s[3]; uint8_t scratch[8]; char name[CAPTURE_NAME_CAPACITY];
    unsigned stage;
    init_three(s,&fs);
    {
        capture_store alias=s[1]; strcpy(alias.directory,"0:/gReY");
        assert(capture_store_save_pair(&s[1],&alias,2,2,rgb_row,&fs,rgb_row,&fs,scratch,8,name)==CAPTURE_INVALID);
        assert(fs.writes==0);
    }
    assert(capture_store_save_pair(&s[1],&s[2],2,2,rgb_row,&fs,rgb_row,&fs,scratch,8,name)==CAPTURE_OK);
    assert(!strcmp(name,"000001.bmp"));
    assert(find(&fs,"0:/Grey/000001.bmp") && find(&fs,"0:/Binary/000001.bmp"));
    assert(!find(&fs,"0:/Image/000001.bmp"));
    assert(capture_store_align_sequences(s,3)==CAPTURE_OK && s[0].next_number==2u);
    assert(save(&s[0],&fs,name)==CAPTURE_OK && !strcmp(name,"000002.bmp"));
    init_three(s,&fs);
    assert(s[0].next_number==3u && s[1].next_number==3u && s[2].next_number==3u);
    reset(&fs);
    /* Fail each stream phase, including second-file staging/commit. Neither
     * target is left behind, and numbering remains available after remount. */
    for(stage=0;stage<12u;++stage) {
        int rc;
        init_three(s,&fs);
        if(stage<6u) fs.fail_write=(int)stage+1;
        if(stage==6u) fs.fail_sync=1;
        if(stage==7u) fs.fail_close=1;
        if(stage==8u) fs.fail_rename_at=1;
        if(stage==9u) fs.fail_rename_at=2;
        if(stage==10u) fs.fail_sync_at=2;
        if(stage==11u) fs.fail_close_at=2;
        rc=capture_store_save_pair(&s[1],&s[2],2,2,rgb_row,&fs,rgb_row,&fs,scratch,8,name);
        assert(rc==CAPTURE_FS_IO && !name[0]);
        assert(!find(&fs,"0:/Grey/000001.bmp") && !find(&fs,"0:/Binary/000001.bmp"));
        assert(!find(&fs,"0:/Grey/PHOTO.TMP") && !find(&fs,"0:/Binary/PHOTO.TMP"));
        assert(s[1].next_number==1u && s[2].next_number==1u);
        reset(&fs);
    }
    /* A collision in commit2 must preserve that existing photograph. */
    init_three(s,&fs); fs.directory="0:/Binary"; add_file(&fs,"000001.bmp");
    assert(capture_store_save_pair(&s[1],&s[2],2,2,rgb_row,&fs,rgb_row,&fs,scratch,8,name)==CAPTURE_FS_EXISTS);
    assert(find(&fs,"0:/Binary/000001.bmp")->size==0u && !find(&fs,"0:/Grey/000001.bmp"));
    init_three(s,&fs); assert(s[0].next_number==2u); reset(&fs);
    /* Unrecoverable rollback/power-loss orphan must never reuse its number. */
    init_three(s,&fs); fs.fail_rename_at=2; fs.fail_remove_final=1;
    assert(capture_store_save_pair(&s[1],&s[2],2,2,rgb_row,&fs,rgb_row,&fs,scratch,8,name)==CAPTURE_PAIR_PARTIAL);
    assert(!s[1].ready && !s[2].ready && find(&fs,"0:/Grey/000001.bmp"));
    fs.fail_rename_at=0; fs.fail_remove_final=0; init_three(s,&fs);
    assert(s[0].next_number==2u && s[1].next_number==2u && s[2].next_number==2u);
    reset(&fs);
    puts("capture_store: paired staging/commit/rollback, collision preservation and three-directory restart numbering PASS");
}

static void test_sequence(void)
{
    memory_fs fs = {0};
    capture_store store, rebooted;
    capture_fs_ops without_cursor_reader = operations;
    memory_file *file;
    char name[CAPTURE_NAME_CAPACITY];
    unsigned stage;
    int writes;
    static const uint8_t expected_pixels[16] = {
        255, 0, 0, 255, 255, 255, 0, 0,
        0, 0, 255, 0, 255, 0, 0, 0
    };
    fs.directory = "0:/Images";
    without_cursor_reader.read_small = NULL;
    assert(CAPTURE_NAME_CAPACITY >= 11);
    assert(capture_store_init_sequence(&store, &without_cursor_reader, &fs, "0:/Images/") == CAPTURE_OK);
    assert(store.mode == CAPTURE_NUMBERING_SEQUENCE && store.next_number == 1);
    assert(capture_store_current_name(&store, name) == CAPTURE_OK && strcmp(name, "000001.bmp") == 0);
    assert(capture_store_next_song(&store) == CAPTURE_INVALID && fs.writes == 0);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "000001.bmp") == 0);
    file = find(&fs, "0:/Images/000001.bmp");
    assert(file && file->size == 70 && file->bytes[0] == 'B' && file->bytes[1] == 'M');
    assert(le32(file->bytes + 2) == 70 && le32(file->bytes + 10) == 54);
    assert(le32(file->bytes + 18) == 2 && le32(file->bytes + 22) == 2 && file->bytes[28] == 24);
    assert(memcmp(file->bytes + 54, expected_pixels, sizeof(expected_pixels)) == 0);
    assert(store.next_number == 2 && store.next_page == 0 && fs.reads == 0);
    assert(!find(&fs, "0:/Images/PHOTO.TMP") && !find(&fs, "0:/Images/CAP0.DAT") && !find(&fs, "0:/Images/CAP1.DAT"));
    assert(capture_store_init_sequence(&rebooted, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(rebooted.next_number == 2 && fs.reads == 0);

    /* Resume at the highest valid filename, ignoring holes and other schemes.
     * Even corrupt old song journals cannot affect sequence initialization. */
    reset(&fs); fs.directory = "0:/Images";
    add_file(&fs, "000002.bmp"); add_file(&fs, "000098.BmP");
    add_file(&fs, "000000.bmp"); add_file(&fs, "999999.txt");
    add_file(&fs, "1000000.bmp"); add_file(&fs, "00099.bmp");
    add_file(&fs, "12345x.bmp"); add_file(&fs, "99-99.bmp");
    add_file(&fs, "CAP0.DAT"); add_file(&fs, "CAP1.DAT");
    assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(store.next_number == 99 && fs.reads == 0 && fs.writes == 0);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "000099.bmp") == 0);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "000100.bmp") == 0);
    assert(capture_store_init_sequence(&rebooted, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(rebooted.next_number == 101 && fs.reads == 0);
    assert(find(&fs, "0:/Images/CAP0.DAT")->size == 0 && find(&fs, "0:/Images/CAP1.DAT")->size == 0);

    /* A directory or a late file is also an occupied name, including mixed case. */
    reset(&fs); fs.directory = "0:/Images";
    add_directory(&fs, "000123.BMP");
    assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(store.next_number == 124);
    add_file(&fs, "000124.BMP");
    assert(save(&store, &fs, name) == CAPTURE_FS_EXISTS && store.next_number == 124 && name[0] == '\0');
    assert(find(&fs, "0:/Images/000124.BMP")->size == 0 && !find(&fs, "0:/Images/PHOTO.TMP"));
    assert(capture_store_init_sequence(&rebooted, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(rebooted.next_number == 125);
    add_directory(&fs, "000125.bmp");
    assert(save(&rebooted, &fs, name) == CAPTURE_FS_EXISTS && rebooted.next_number == 125);
    assert(find(&fs, "0:/Images/000125.bmp")->is_directory);

    /* Every failed write stage leaves the same number available for retry. */
    for (stage = 0; stage < 7; ++stage) {
        int expected = CAPTURE_FS_IO;
        reset(&fs); fs.directory = "0:/Images";
        add_file(&fs, "000040.bmp");
        assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_OK);
        switch (stage) {
        case 0: fs.fail_write = 1; break;
        case 1: fs.fail_write = 3; break;
        case 2: fs.short_write = 2; expected = CAPTURE_SHORT_WRITE; break;
        case 3: fs.fail_sync = 1; break;
        case 4: fs.fail_close = 1; break;
        case 5: fs.fail_rename = 1; break;
        case 6: fs.fail_row = 1; expected = CAPTURE_ROW_ERROR; break;
        }
        assert(save(&store, &fs, name) == expected && store.next_number == 41 && name[0] == '\0');
        assert(!find(&fs, "0:/Images/000041.bmp") && !find(&fs, "0:/Images/PHOTO.TMP"));
        fs.fail_write = fs.short_write = fs.fail_sync = fs.fail_close = fs.fail_rename = fs.fail_row = 0;
        assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "000041.bmp") == 0);
        assert(store.next_number == 42 && fs.reads == 0);
    }

    /* The last number can be committed once; neither RAM nor reboot wraps. */
    reset(&fs); fs.directory = "0:/Images";
    add_file(&fs, "999998.bmp");
    assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "999999.bmp") == 0);
    assert(store.next_number == 1000000u);
    writes = fs.writes;
    assert(capture_store_current_name(&store, name) == CAPTURE_NUMBER_FULL);
    assert(save(&store, &fs, name) == CAPTURE_NUMBER_FULL && name[0] == '\0' && fs.writes == writes);
    assert(capture_store_init_sequence(&rebooted, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(rebooted.next_number == 1000000u && save(&rebooted, &fs, name) == CAPTURE_NUMBER_FULL);
    reset(&fs); fs.directory = "0:/Images";
    add_directory(&fs, "999999.BmP");
    assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_OK);
    assert(capture_store_current_name(&store, name) == CAPTURE_NUMBER_FULL && fs.writes == 0);

    reset(&fs); fs.directory = "0:/Images"; fs.offline = 1;
    assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_FS_IO);
    assert(save(&store, &fs, name) == CAPTURE_NOT_READY);
    fs.offline = 0;
    assert(capture_store_init_sequence(&store, &operations, &fs, "0:/Images") == CAPTURE_OK);
    fs.offline = 1;
    assert(save(&store, &fs, name) == CAPTURE_FS_IO && store.next_number == 1);
    fs.offline = 0;
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "000001.bmp") == 0);
    reset(&fs);
}

static int indexed_row(void *context,uint32_t y,uint8_t *data,size_t length)
{
    unsigned bpp=*(unsigned *)context;
    size_t x;
    for(x=0;x<length;++x) data[x]=bpp==1u?(((x+y)%3u)?255u:0u):(uint8_t)(x+y);
    return CAPTURE_OK;
}

static int invalid_binary_row(void *context,uint32_t y,uint8_t *data,size_t length)
{
    (void)context;(void)y;memset(data,127,length);return CAPTURE_OK;
}

static void test_indexed(void)
{
    const unsigned widths[]={1,7,8,9,31,32,33,257,4056};
    unsigned bpp,i,x,y;
    uint8_t scratch[4056];
    char name[CAPTURE_NAME_CAPACITY];
    for(bpp=1;bpp<=8;bpp+=7) for(i=0;i<sizeof(widths)/sizeof(widths[0]);++i) {
        memory_fs fs={0};capture_store store;memory_file *f;
        unsigned w=widths[i],offset=bpp==1?62u:1078u,row=((w*bpp+31u)/32u)*4u;
        assert(capture_store_init_sequence(&store,&operations,&fs,"0:/Image")==CAPTURE_OK);
        assert(capture_store_save_bmp_format(&store,w,3,bpp,indexed_row,&bpp,
            scratch,sizeof(scratch),name)==CAPTURE_OK);
        f=find(&fs,"0:/Image/000001.bmp");assert(f);
        assert(f->size==offset+row*3u && le32(f->bytes+10)==offset && f->bytes[28]==bpp);
        assert(le32(f->bytes+2)==f->size && le32(f->bytes+34)==row*3u);
        for(x=0;x<(bpp==1?2u:256u);++x) {
            unsigned v=bpp==1?255u*x:x;
            assert(f->bytes[54+4*x]==v && f->bytes[55+4*x]==v && f->bytes[56+4*x]==v && !f->bytes[57+4*x]);
        }
        for(y=0;y<3;++y) for(x=0;x<row*(bpp==1?8u:1u);++x) {
            unsigned got=bpp==1?((f->bytes[offset+y*row+x/8]>>(7u-x%8))&1u)*255u:f->bytes[offset+y*row+x];
            unsigned want=x<w?(bpp==1?(((x+2u-y)%3)?255u:0u):(uint8_t)(x+2u-y)):0u;
            assert(got==want);
        }
        assert(capture_store_save_bmp_format(&store,w,3,1,invalid_binary_row,NULL,
            scratch,sizeof(scratch),name)==CAPTURE_ROW_ERROR);
        assert(store.next_number==2 && !find(&fs,"0:/Image/PHOTO.TMP"));reset(&fs);
    }
    {
        memory_fs fs={0};capture_store stores[2];memory_file *f;FILE *out;
        unsigned gray=8,binary=1;
        fs.multiple_directories=1;
        assert(capture_store_init_sequence(stores,&operations,&fs,"0:/Grey")==CAPTURE_OK);
        assert(capture_store_init_sequence(stores+1,&operations,&fs,"0:/Binary")==CAPTURE_OK);
        assert(capture_store_save_pair_formats(stores,stores+1,4056,2868,8,indexed_row,&gray,
            1,indexed_row,&binary,scratch,sizeof(scratch),name)==CAPTURE_OK);
        f=find(&fs,"0:/Grey/000001.bmp");assert(f && f->size==11633686u);
        out=fopen("build/camera_capture_tests/gray8.bmp","wb");assert(out);
        assert(fwrite(f->bytes,1,f->size,out)==f->size);assert(fclose(out)==0);
        f=find(&fs,"0:/Binary/000001.bmp");assert(f && f->size==1457006u);
        out=fopen("build/camera_capture_tests/binary1.bmp","wb");assert(out);
        assert(fwrite(f->bytes,1,f->size,out)==f->size);assert(fclose(out)==0);
        fs.fail_rename_at=fs.rename_calls+2;
        assert(capture_store_save_pair_formats(stores,stores+1,4056,2,8,indexed_row,&gray,
            1,indexed_row,&binary,scratch,sizeof(scratch),name)==CAPTURE_FS_IO);
        assert(stores[0].next_number==2 && !find(&fs,"0:/Grey/000002.bmp") && !find(&fs,"0:/Binary/000002.bmp"));
        reset(&fs);
    }
    puts("PASS indexed BMP: all gray codes, bit order, padding, invalid input, native pair sizes, rollback, decoder fixtures");
}

static const char *const album_paths[3]={"0:/Image","0:/Grey","0:/Meta"};
static const uint8_t album_json[]="{\"complete\":true,\"same_frame\":true}\n";

static int album_save(capture_store stores[3],memory_fs *fs,char *name)
{
    unsigned format=8u;
    uint8_t scratch[8];
    return capture_store_save_album(stores,2,2,rgb_row,fs,indexed_row,&format,
        album_json,sizeof(album_json)-1u,scratch,sizeof(scratch),name);
}

static void album_init(capture_store stores[3],memory_fs *fs)
{
    fs->multiple_directories=1;
    assert(capture_store_init_album(stores,&operations,fs,album_paths)==CAPTURE_OK);
}

static void test_album(void)
{
    memory_fs fs={0};
    capture_store stores[3],rebooted[3];
    char name[CAPTURE_NAME_CAPACITY];
    unsigned stage;
    album_init(stores,&fs);
    assert(album_save(stores,&fs,name)==CAPTURE_OK && strcmp(name,"00-00.bmp")==0);
    assert(find(&fs,"0:/Image/00-00.bmp")->bytes[28]==24u);
    assert(find(&fs,"0:/Grey/00-00.bmp")->bytes[28]==8u);
    assert(find(&fs,"0:/Meta/00-00.json")->size==sizeof(album_json)-1u);
    assert(fs.rename_calls==3 && !find(&fs,"0:/Binary/00-00.bmp"));
    album_init(rebooted,&fs);
    assert(rebooted[0].song==0 && rebooted[0].next_page==1);
    assert(capture_store_album_next_song(rebooted)==CAPTURE_OK);
    album_init(stores,&fs);
    assert(stores[0].song==1 && stores[1].song==1 && stores[2].next_page==0);
    assert(album_save(stores,&fs,name)==CAPTURE_OK && strcmp(name,"01-00.bmp")==0);

    reset(&fs);fs.multiple_directories=1;
    fs.directory="0:/Image";add_file(&fs,"999999.bmp");add_file(&fs,"02-01.bmp");
    fs.directory="0:/Grey";add_file(&fs,"02-03.BMP");
    fs.directory="0:/Meta";add_file(&fs,"02-05.JSON");add_directory(&fs,"02-06.json");
    album_init(stores,&fs);
    assert(stores[0].song==2 && stores[0].next_page==7 && stores[2].next_page==7);

    /* Every stage is covered: gray/metadata failure must not commit color;
     * the JSON completion marker is renamed after both image commits. */
    for(stage=0;stage<13u;++stage) {
        int expected=CAPTURE_FS_IO;
        reset(&fs);album_init(stores,&fs);
        switch(stage) {
        case 0:fs.fail_write=1;break;
        case 1:fs.fail_write=4;break;
        case 2:fs.fail_write=7;break;
        case 3:fs.short_write=7;expected=CAPTURE_SHORT_WRITE;break;
        case 4:fs.fail_sync_at=1;break;
        case 5:fs.fail_sync_at=2;break;
        case 6:fs.fail_sync_at=3;break;
        case 7:fs.fail_close_at=3;break;
        case 8:fs.fail_rename_at=1;break;
        case 9:fs.fail_rename_at=2;break;
        case 10:fs.fail_rename_at=3;break;
        case 11:fs.fail_row=1;expected=CAPTURE_ROW_ERROR;break;
        case 12:fs.short_write=5;expected=CAPTURE_SHORT_WRITE;break;
        }
        assert(album_save(stores,&fs,name)==expected && name[0]=='\0');
        assert(stores[0].next_page==0 && stores[1].next_page==0 && stores[2].next_page==0);
        assert(!find(&fs,"0:/Image/00-00.bmp") && !find(&fs,"0:/Grey/00-00.bmp") && !find(&fs,"0:/Meta/00-00.json"));
        assert(!find(&fs,"0:/Image/PHOTO.TMP") && !find(&fs,"0:/Grey/PHOTO.TMP") && !find(&fs,"0:/Meta/PHOTO.TMP"));
    }
    reset(&fs);album_init(stores,&fs);
    fs.directory="0:/Meta";add_file(&fs,"00-00.json");
    assert(album_save(stores,&fs,name)==CAPTURE_FS_EXISTS);
    assert(find(&fs,"0:/Meta/00-00.json")->size==0 && !find(&fs,"0:/Image/00-00.bmp"));
    album_init(rebooted,&fs);assert(rebooted[0].next_page==1);

    reset(&fs);album_init(stores,&fs);
    fs.fail_rename_at=3;fs.fail_remove_final=1;
    assert(album_save(stores,&fs,name)==CAPTURE_PAIR_PARTIAL);
    assert(!stores[0].ready && !stores[1].ready && !stores[2].ready);
    fs.fail_remove_final=0;fs.fail_rename_at=0;
    album_init(rebooted,&fs);assert(rebooted[0].next_page==1);
    assert(album_save(rebooted,&fs,name)==CAPTURE_OK && strcmp(name,"00-01.bmp")==0);

    reset(&fs);album_init(stores,&fs);
    fs.fail_sync_at=1;
    assert(capture_store_album_next_song(stores)==CAPTURE_FS_IO && stores[0].song==0);
    fs.fail_sync_at=0;album_init(rebooted,&fs);
    assert(rebooted[0].song==0);
    reset(&fs);fs.multiple_directories=1;fs.directory="0:/Grey";add_file(&fs,"98-99.bmp");
    album_init(stores,&fs);assert(album_save(stores,&fs,name)==CAPTURE_PAGE_FULL);
    assert(capture_store_album_next_song(stores)==CAPTURE_OK && stores[0].song==99);
    fs.directory="0:/Meta";add_file(&fs,"99-99.json");
    album_init(stores,&fs);assert(stores[0].next_page==100);
    assert(capture_store_album_next_song(stores)==CAPTURE_SONG_FULL);

    reset(&fs);album_init(stores,&fs);
    {
        unsigned format=8u;
        uint8_t *scratch=malloc(4056u*3u);
        full_frame frame={0,0};
        assert(scratch);
        assert(capture_store_save_album(stores,4056,2868,full_rgb_row,&frame,indexed_row,&format,
            album_json,sizeof(album_json)-1u,scratch,4056u*3u,name)==CAPTURE_OK);
        assert(find(&fs,"0:/Image/00-00.bmp")->size==34897878u);
        assert(find(&fs,"0:/Grey/00-00.bmp")->size==11633686u);
        free(scratch);
    }
    reset(&fs);
    puts("capture_store album: native RGB+gray+JSON, restart/orphan/collision, empty-song journal, limits and failure rollback PASS");
}

int main(void)
{
    memory_fs fs = {0};
    capture_store store, rebooted;
    memory_file *f;
    char name[CAPTURE_NAME_CAPACITY];
    uint8_t row[8];
    unsigned stage;
    static const uint8_t expected_pixels[16] = {
        255, 0, 0, 255, 255, 255, 0, 0, /* bottom blue, white, padding */
        0, 0, 255, 0, 255, 0, 0, 0     /* top red, green, padding */
    };

    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(capture_store_current_name(&store, name) == CAPTURE_OK && strcmp(name, "00-00.bmp") == 0);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "00-00.bmp") == 0);
    f = find(&fs, "0:/Image/00-00.bmp");
    assert(f && f->size == 70 && f->bytes[0] == 'B' && f->bytes[1] == 'M');
    assert(le32(f->bytes + 2) == 70 && le32(f->bytes + 10) == 54);
    assert(le32(f->bytes + 14) == 40 && le32(f->bytes + 18) == 2 && le32(f->bytes + 22) == 2);
    assert(f->bytes[26] == 1 && f->bytes[28] == 24 && le32(f->bytes + 34) == 16);
    assert(memcmp(f->bytes + 54, expected_pixels, 16) == 0);
    assert(!find(&fs, "0:/Image/PHOTO.TMP"));
    assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(rebooted.song == 0 && rebooted.next_page == 1);
    assert(save(&rebooted, &fs, name) == CAPTURE_OK && strcmp(name, "00-01.bmp") == 0);

    /* Changing song persists even before a photograph exists in that song. */
    assert(capture_store_next_song(&rebooted) == CAPTURE_OK);
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(store.song == 1 && store.next_page == 0);
    assert(capture_store_next_song(&store) == CAPTURE_OK);
    assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(rebooted.song == 2 && rebooted.next_page == 0);
    assert(find(&fs, "0:/Image/CAP0.DAT") && find(&fs, "0:/Image/CAP1.DAT"));

    /* Partial journal updates leave the preceding committed cursor readable. */
    fs.fail_rename = 1;
    assert(capture_store_next_song(&rebooted) == CAPTURE_FS_IO && rebooted.song == 2);
    fs.fail_rename = 0;
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(store.song == 2 && store.next_page == 0);
    assert(capture_store_next_song(&store) == CAPTURE_OK && store.song == 3);
    f = find(&fs, "0:/Image/CAP1.DAT");
    assert(f); f->bytes[0] ^= 1;
    assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(rebooted.song == 2); /* intact older slot */
    find(&fs, "0:/Image/CAP0.DAT")->bytes[0] ^= 1;
    assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_STATE_CORRUPT);
    assert(!rebooted.ready);

    /* An oversized torn newer journal must not hide the intact older slot. */
    reset(&fs);
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(capture_store_next_song(&store) == CAPTURE_OK);
    assert(capture_store_next_song(&store) == CAPTURE_OK);
    f = find(&fs, "0:/Image/CAP0.DAT");
    assert(f && f->capacity > f->size);
    ++f->size;
    assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(rebooted.song == 1);

    reset(&fs);
    add_file(&fs, "00-02.bmp"); add_file(&fs, "09-13.BMP"); add_file(&fs, "09-02.bmp");
    add_file(&fs, "99-99.txt"); add_file(&fs, "nonsense.bmp");
    add_directory(&fs, "99-99.bmp"); /* Legacy song mode still ignores directories. */
    assert(capture_store_init(&store, &operations, &fs, "0:/Image/") == CAPTURE_OK);
    assert(store.song == 9 && store.next_page == 14);
    /* Unexpected late collision cannot overwrite an existing picture. */
    add_file(&fs, "09-14.bmp");
    assert(save(&store, &fs, name) == CAPTURE_FS_EXISTS && store.next_page == 14);
    assert(find(&fs, "0:/Image/09-14.bmp")->size == 0);
    assert(!find(&fs, "0:/Image/PHOTO.TMP"));

    reset(&fs);
    add_file(&fs, "98-99.bmp");
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(save(&store, &fs, name) == CAPTURE_PAGE_FULL && store.song == 98);
    assert(capture_store_next_song(&store) == CAPTURE_OK && store.song == 99);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "99-00.bmp") == 0);
    assert(capture_store_next_song(&store) == CAPTURE_SONG_FULL);
    add_file(&fs, "99-98.bmp");
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "99-99.bmp") == 0);
    assert(save(&store, &fs, name) == CAPTURE_PAGE_FULL);
    assert(capture_store_next_song(&store) == CAPTURE_SONG_FULL);

    /* All write phases fail cleanly, retaining the same next image number. */
    for (stage = 0; stage < 7; ++stage) {
        int expected = CAPTURE_FS_IO;
        reset(&fs);
        assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
        switch (stage) {
        case 0: fs.fail_write = 1; break;
        case 1: fs.fail_write = 3; break;
        case 2: fs.short_write = 2; expected = CAPTURE_SHORT_WRITE; break;
        case 3: fs.fail_sync = 1; break;
        case 4: fs.fail_close = 1; break;
        case 5: fs.fail_rename = 1; break;
        case 6: fs.fail_row = 1; expected = CAPTURE_ROW_ERROR; break;
        }
        assert(save(&store, &fs, name) == expected);
        assert(store.next_page == 0 && name[0] == '\0');
        assert(!find(&fs, "0:/Image/00-00.bmp") && !find(&fs, "0:/Image/PHOTO.TMP"));
        fs.fail_write = fs.short_write = fs.fail_sync = fs.fail_close = fs.fail_rename = fs.fail_row = 0;
        assert(save(&store, &fs, name) == CAPTURE_OK && strcmp(name, "00-00.bmp") == 0);
    }

    /* Failed song changes never acknowledge an uncommitted empty song. */
    for (stage = 0; stage < 5; ++stage) {
        int expected = CAPTURE_FS_IO;
        reset(&fs);
        assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
        assert(capture_store_next_song(&store) == CAPTURE_OK && store.song == 1);
        switch (stage) {
        case 0: fs.fail_write = fs.writes + 1; break;
        case 1: fs.short_write = fs.writes + 1; expected = CAPTURE_SHORT_WRITE; break;
        case 2: fs.fail_sync = 1; break;
        case 3: fs.fail_close = 1; break;
        case 4: fs.fail_rename = 1; break;
        }
        assert(capture_store_next_song(&store) == expected && store.song == 1);
        fs.fail_write = fs.short_write = fs.fail_sync = fs.fail_close = fs.fail_rename = 0;
        assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_OK);
        assert(rebooted.song == 1 && rebooted.next_page == 0);
    }

    reset(&fs); fs.offline = 1;
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_FS_IO);
    assert(save(&store, &fs, name) == CAPTURE_NOT_READY);
    fs.offline = 0;
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    fs.offline = 1;
    assert(save(&store, &fs, name) == CAPTURE_FS_IO && store.next_page == 0);
    assert(capture_store_next_song(&store) == CAPTURE_FS_IO && store.song == 0);
    fs.offline = 0;
    assert(capture_store_save_bmp(&store, 2, 2, rgb_row, &fs, row, 7, name) == CAPTURE_INVALID);
    assert(capture_store_save_bmp(&store, 0, 2, rgb_row, &fs, row, 8, name) == CAPTURE_INVALID);
    assert(capture_store_save_bmp(&store, UINT32_MAX, UINT32_MAX, rgb_row, &fs, row, 8, name) == CAPTURE_INVALID);
    assert(capture_store_next_song(&store) == CAPTURE_OK);
    fs.fail_write = fs.writes + 1;
    assert(capture_store_next_song(&store) == CAPTURE_FS_IO && store.song == 1);
    fs.fail_write = 0;
    assert(capture_store_init(&rebooted, &operations, &fs, "0:/Image") == CAPTURE_OK);
    assert(rebooted.song == 1);

    /* A complete HQ-sized frame is streamed with only a 12,168-byte caller row.
     * The fake filesystem keeps output bytes solely for validation. */
    reset(&fs);
    assert(capture_store_init(&store, &operations, &fs, "0:/Image") == CAPTURE_OK);
    {
        uint8_t *scratch = malloc(4056u * 3u);
        full_frame frame = {0, 0};
        assert(scratch);
        assert(capture_store_save_bmp(&store, 4056, 3040, full_rgb_row, &frame,
                                     scratch, 4056u * 3u, name) == CAPTURE_OK);
        assert(frame.calls == 3040 && frame.last_y == 0);
        f = find(&fs, "0:/Image/00-00.bmp");
        assert(f && f->size == 36990774u && le32(f->bytes + 2) == f->size);
        assert(le32(f->bytes + 18) == 4056 && le32(f->bytes + 22) == 3040);
        assert(f->bytes[54] == 127 && f->bytes[56] == (uint8_t)3039);
        assert(f->bytes[f->size - 3u] == 127 && f->bytes[f->size - 1u] == 0);
        free(scratch);
    }
    reset(&fs);
    test_sequence();
    test_pair();
    test_indexed();
    test_album();
    puts("capture_store: BMP bytes, song journal, sequence resume/collision/full and failure cases passed");
    return 0;
}
