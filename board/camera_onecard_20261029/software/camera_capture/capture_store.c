#include "capture_store.h"

#include <limits.h>
#include <ctype.h>
#include <string.h>

#define CURSOR_SIZE 20u
#define CURSOR_MAGIC 0x31504143u /* little-endian "CAP1" */
#define MAX_CAPTURE_NUMBER 999999u

static void put32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static uint32_t get32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint32_t crc32(const uint8_t *data, size_t length)
{
    uint32_t crc = UINT32_MAX;
    size_t i;
    unsigned bit;
    for (i = 0; i < length; ++i) {
        crc ^= data[i];
        for (bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static int path_for(const capture_store *s, const char *name,
                    char path[CAPTURE_PATH_CAPACITY])
{
    size_t directory_length = strlen(s->directory), name_length = strlen(name);
    if (directory_length + name_length + 2u > CAPTURE_PATH_CAPACITY)
        return CAPTURE_INVALID;
    memcpy(path, s->directory, directory_length);
    path[directory_length] = '/';
    memcpy(path + directory_length + 1u, name, name_length + 1u);
    return CAPTURE_OK;
}

static int remove_if_present(capture_store *s, const char *path)
{
    int result = s->fs->remove(s->fs_context, path);
    return result == CAPTURE_FS_NOENT ? CAPTURE_OK : result;
}

static int write_all(capture_store *s, void *file, const uint8_t *data, size_t length)
{
    size_t written = 0;
    int result = s->fs->write(s->fs_context, file, data, length, &written);
    if (result != CAPTURE_OK)
        return result;
    return written == length ? CAPTURE_OK : CAPTURE_SHORT_WRITE;
}

typedef struct {
    int pages[100];
    int newest_song;
} scan_state;

static int scan_entry(void *context, const char *name, int is_directory)
{
    scan_state *scan = (scan_state *)context;
    unsigned song, page;
    if (is_directory || strlen(name) != 9 || name[2] != '-' || name[5] != '.' ||
        (name[6] != 'b' && name[6] != 'B') ||
        (name[7] != 'm' && name[7] != 'M') ||
        (name[8] != 'p' && name[8] != 'P') ||
        name[0] < '0' || name[0] > '9' || name[1] < '0' || name[1] > '9' ||
        name[3] < '0' || name[3] > '9' || name[4] < '0' || name[4] > '9')
        return CAPTURE_OK;
    song = (unsigned)(name[0] - '0') * 10u + (unsigned)(name[1] - '0');
    page = (unsigned)(name[3] - '0') * 10u + (unsigned)(name[4] - '0');
    if ((int)page > scan->pages[song])
        scan->pages[song] = (int)page;
    if ((int)song > scan->newest_song)
        scan->newest_song = (int)song;
    return CAPTURE_OK;
}

int capture_store_init(capture_store *s, const capture_fs_ops *fs,
                       void *fs_context, const char *directory)
{
    scan_state scan;
    char path[CAPTURE_PATH_CAPACITY];
    uint8_t record[CURSOR_SIZE];
    unsigned slot, i, valid = 0, invalid = 0;
    size_t length;
    int result;
    if (!s)
        return CAPTURE_INVALID;
    memset(s, 0, sizeof(*s));
    if (!fs || !directory || !*directory || strlen(directory) + 12u >= sizeof(s->directory) ||
        !fs->mkdir || !fs->list || !fs->read_small || !fs->create_exclusive ||
        !fs->write || !fs->sync || !fs->close || !fs->rename_no_replace || !fs->remove)
        return CAPTURE_INVALID;
    s->fs = fs;
    s->fs_context = fs_context;
    strcpy(s->directory, directory);
    while (strlen(s->directory) > 1u && s->directory[strlen(s->directory) - 1u] == '/')
        s->directory[strlen(s->directory) - 1u] = '\0';
    result = fs->mkdir(fs_context, s->directory);
    if (result != CAPTURE_OK && result != CAPTURE_FS_EXISTS)
        return result;
    for (i = 0; i < 100u; ++i)
        scan.pages[i] = -1;
    scan.newest_song = -1;
    result = fs->list(fs_context, s->directory, scan_entry, &scan);
    if (result != CAPTURE_OK)
        return result;
    for (slot = 0; slot < 2u; ++slot) {
        path_for(s, slot ? "CAP1.DAT" : "CAP0.DAT", path);
        length = 0;
        result = fs->read_small(fs_context, path, record, sizeof(record), &length);
        if (result == CAPTURE_FS_NOENT)
            continue;
        if (result == CAPTURE_STATE_CORRUPT) {
            ++invalid;
            continue;
        }
        if (result != CAPTURE_OK)
            return result;
        if (length != CURSOR_SIZE || get32(record) != CURSOR_MAGIC ||
            get32(record + 12) != 0 || get32(record + 8) > 99u ||
            crc32(record, 16) != get32(record + 16)) {
            ++invalid;
            continue;
        }
        if (!valid || get32(record + 4) > s->cursor_sequence) {
            s->cursor_sequence = get32(record + 4);
            s->song = get32(record + 8);
        }
        ++valid;
    }
    if (invalid && !valid)
        return CAPTURE_STATE_CORRUPT;
    if (scan.newest_song > (int)s->song)
        s->song = (unsigned)scan.newest_song;
    s->next_page = (unsigned)(scan.pages[s->song] + 1);
    s->ready = 1;
    return CAPTURE_OK;
}

static int scan_sequence_entry(void *context, const char *name, int is_directory)
{
    unsigned *maximum = context;
    unsigned number = 0, i;
    /* Directories with image-like names occupy a destination just as files do. */
    (void)is_directory;
    if (strlen(name) != 10 || name[6] != '.' ||
        (name[7] != 'b' && name[7] != 'B') ||
        (name[8] != 'm' && name[8] != 'M') ||
        (name[9] != 'p' && name[9] != 'P'))
        return CAPTURE_OK;
    for (i = 0; i < 6u; ++i) {
        if (name[i] < '0' || name[i] > '9')
            return CAPTURE_OK;
        number = number * 10u + (unsigned)(name[i] - '0');
    }
    if (number > *maximum)
        *maximum = number;
    return CAPTURE_OK;
}

int capture_store_init_sequence(capture_store *s, const capture_fs_ops *fs,
                                void *fs_context, const char *directory)
{
    unsigned maximum = 0;
    int result;
    if (!s)
        return CAPTURE_INVALID;
    memset(s, 0, sizeof(*s));
    if (!fs || !directory || !*directory || strlen(directory) + 12u >= sizeof(s->directory) ||
        !fs->mkdir || !fs->list || !fs->create_exclusive || !fs->write ||
        !fs->sync || !fs->close || !fs->rename_no_replace || !fs->remove)
        return CAPTURE_INVALID;
    s->fs = fs;
    s->fs_context = fs_context;
    s->mode = CAPTURE_NUMBERING_SEQUENCE;
    strcpy(s->directory, directory);
    while (strlen(s->directory) > 1u && s->directory[strlen(s->directory) - 1u] == '/')
        s->directory[strlen(s->directory) - 1u] = '\0';
    result = fs->mkdir(fs_context, s->directory);
    if (result != CAPTURE_OK && result != CAPTURE_FS_EXISTS)
        return result;
    result = fs->list(fs_context, s->directory, scan_sequence_entry, &maximum);
    if (result != CAPTURE_OK)
        return result;
    s->next_number = maximum + 1u;
    s->ready = 1;
    return CAPTURE_OK;
}

int capture_store_current_name(const capture_store *s, char name[CAPTURE_NAME_CAPACITY])
{
    if (!s || !s->ready)
        return CAPTURE_NOT_READY;
    if (!name)
        return CAPTURE_INVALID;
    if (s->mode == CAPTURE_NUMBERING_SEQUENCE) {
        unsigned number = s->next_number, i;
        if (number > MAX_CAPTURE_NUMBER)
            return CAPTURE_NUMBER_FULL;
        if (!number)
            return CAPTURE_INVALID;
        for (i = 6u; i > 0; --i) {
            name[i - 1u] = (char)('0' + number % 10u);
            number /= 10u;
        }
        memcpy(name + 6, ".bmp", 5);
        return CAPTURE_OK;
    }
    if (s->mode != CAPTURE_NUMBERING_SONG_PAGE)
        return CAPTURE_INVALID;
    if (s->next_page > 99u)
        return CAPTURE_PAGE_FULL;
    if (s->song > 99u)
        return CAPTURE_INVALID;
    name[0] = (char)('0' + s->song / 10u);
    name[1] = (char)('0' + s->song % 10u);
    name[2] = '-';
    name[3] = (char)('0' + s->next_page / 10u);
    name[4] = (char)('0' + s->next_page % 10u);
    memcpy(name + 5, ".bmp", 5);
    return CAPTURE_OK;
}

int capture_store_next_song(capture_store *s)
{
    uint8_t record[CURSOR_SIZE] = {0};
    char temporary[CAPTURE_PATH_CAPACITY], target[CAPTURE_PATH_CAPACITY];
    void *file = NULL;
    uint32_t sequence;
    int result, close_result;
    if (!s || !s->ready)
        return CAPTURE_NOT_READY;
    if (s->mode != CAPTURE_NUMBERING_SONG_PAGE)
        return CAPTURE_INVALID;
    if (s->song >= 99u)
        return CAPTURE_SONG_FULL;
    if (s->cursor_sequence == UINT32_MAX)
        return CAPTURE_STATE_CORRUPT;
    sequence = s->cursor_sequence + 1u;
    path_for(s, sequence & 1u ? "CAP1.TMP" : "CAP0.TMP", temporary);
    path_for(s, sequence & 1u ? "CAP1.DAT" : "CAP0.DAT", target);
    result = remove_if_present(s, temporary);
    if (result != CAPTURE_OK)
        return result;
    result = s->fs->create_exclusive(s->fs_context, temporary, &file);
    if (result != CAPTURE_OK)
        return result;
    put32(record, CURSOR_MAGIC);
    put32(record + 4, sequence);
    put32(record + 8, s->song + 1u);
    put32(record + 16, crc32(record, 16));
    result = write_all(s, file, record, sizeof(record));
    if (result == CAPTURE_OK)
        result = s->fs->sync(s->fs_context, file);
    close_result = s->fs->close(s->fs_context, file);
    if (result == CAPTURE_OK)
        result = close_result;
    if (result == CAPTURE_OK)
        result = remove_if_present(s, target);
    if (result == CAPTURE_OK)
        result = s->fs->rename_no_replace(s->fs_context, temporary, target);
    if (result != CAPTURE_OK) {
        remove_if_present(s, temporary);
        return result;
    }
    s->cursor_sequence = sequence;
    ++s->song;
    s->next_page = 0;
    return CAPTURE_OK;
}

static int stage_bmp(capture_store *s, uint32_t width, uint32_t height,
                           unsigned format, capture_read_rgb_row read_row, void *row_context,
                           uint8_t *scratch, size_t capacity)
{
    uint8_t header[1078] = {0};
    char name[CAPTURE_NAME_CAPACITY], temporary[CAPTURE_PATH_CAPACITY];
    uint64_t pixels, packed, row_size, image_size;
    unsigned palette, header_size;
    uint32_t y;
    size_t x;
    void *file = NULL;
    int result, close_result;
    result = capture_store_current_name(s, name);
    if (result != CAPTURE_OK)
        return result;
    if(format!=24u && format!=8u && format!=1u) return CAPTURE_INVALID;
    palette=format==8u?256u:(format==1u?2u:0u);
    header_size=54u+palette*4u;
    pixels = (uint64_t)width * (format==24u?3u:1u);
    packed=((uint64_t)width*format+7u)/8u;
    row_size = (packed + 3u) & ~(uint64_t)3u;
    image_size = row_size * height;
    if (!width || !height || width > INT32_MAX || height > INT32_MAX ||
        image_size > UINT32_MAX - header_size || row_size > capacity || pixels > capacity ||
        !scratch || !read_row)
        return CAPTURE_INVALID;
    path_for(s, "PHOTO.TMP", temporary);
    result = remove_if_present(s, temporary);
    if (result != CAPTURE_OK)
        return result;
    result = s->fs->create_exclusive(s->fs_context, temporary, &file);
    if (result != CAPTURE_OK)
        return result;
    header[0] = 'B'; header[1] = 'M';
    put32(header + 2, (uint32_t)image_size + header_size);
    put32(header + 10, header_size);
    put32(header + 14, 40u);
    put32(header + 18, width);
    put32(header + 22, height);
    header[26] = 1; header[28] = (uint8_t)format;
    put32(header + 34, (uint32_t)image_size);
    put32(header + 46, palette);
    for(x=0;x<palette;++x) {
        uint8_t value=(uint8_t)(format==1u?255u*x:x);
        header[54u+4u*x]=header[55u+4u*x]=header[56u+4u*x]=value;
    }
    if(s->fs->reserve) result=s->fs->reserve(s->fs_context,file,(uint32_t)image_size+header_size);
    if(result==CAPTURE_OK) result = write_all(s, file, header, header_size);
    for (y = height; result == CAPTURE_OK && y > 0u; --y) {
        if (read_row(row_context, y - 1u, scratch, (size_t)pixels) != CAPTURE_OK) {
            result = CAPTURE_ROW_ERROR;
            break;
        }
        if(format==24u) for (x = 0; x < (size_t)pixels; x += 3u) {
            uint8_t red = scratch[x];
            scratch[x] = scratch[x + 2u];
            scratch[x + 2u] = red;
        }
        if(format==1u) {
            uint8_t bits=0;
            for(x=0;x<width;++x) {
                if(scratch[x]!=0u && scratch[x]!=255u) { result=CAPTURE_ROW_ERROR; break; }
                if(scratch[x]) bits|=(uint8_t)(0x80u>>(x&7u));
                /* Each source group is consumed before its packed byte is written. */
                if((x&7u)==7u || x+1u==width) { scratch[x/8u]=bits;bits=0; }
            }
            if(result!=CAPTURE_OK) break;
        }
        memset(scratch + (size_t)packed, 0, (size_t)(row_size - packed));
        result = write_all(s, file, scratch, (size_t)row_size);
    }
    if (result == CAPTURE_OK)
        result = s->fs->sync(s->fs_context, file);
    close_result = s->fs->close(s->fs_context, file);
    if (result == CAPTURE_OK)
        result = close_result;
    if (result != CAPTURE_OK) {
        remove_if_present(s, temporary);
        return result;
    }
    return CAPTURE_OK;
}

static int commit_bmp(capture_store *s, const char *name)
{
    char temporary[CAPTURE_PATH_CAPACITY], target[CAPTURE_PATH_CAPACITY];
    path_for(s,"PHOTO.TMP",temporary); path_for(s,name,target);
    return s->fs->rename_no_replace(s->fs_context,temporary,target);
}

static void discard_bmp(capture_store *s)
{
    char temporary[CAPTURE_PATH_CAPACITY];
    path_for(s,"PHOTO.TMP",temporary); (void)remove_if_present(s,temporary);
}

int capture_store_save_bmp_format(capture_store *s, uint32_t width, uint32_t height,
                           unsigned format, capture_read_rgb_row read_row, void *row_context,
                           uint8_t *scratch, size_t capacity,
                           char saved_name[CAPTURE_NAME_CAPACITY])
{
    char name[CAPTURE_NAME_CAPACITY];
    int result;
    if(saved_name) saved_name[0]='\0';
    result=capture_store_current_name(s,name);
    if(result!=CAPTURE_OK) return result;
    result=stage_bmp(s,width,height,format,read_row,row_context,scratch,capacity);
    if(result!=CAPTURE_OK) return result;
    result=commit_bmp(s,name);
    if(result!=CAPTURE_OK) { discard_bmp(s); return result; }
    if (s->mode == CAPTURE_NUMBERING_SEQUENCE)
        ++s->next_number;
    else
        ++s->next_page;
    if (saved_name)
        strcpy(saved_name, name);
    return CAPTURE_OK;
}

int capture_store_align_sequences(capture_store *stores, size_t count)
{
    unsigned maximum=1u;
    size_t i;
    if(!stores || !count) return CAPTURE_INVALID;
    for(i=0;i<count;++i) {
        if(!stores[i].ready) return CAPTURE_NOT_READY;
        if(stores[i].mode!=CAPTURE_NUMBERING_SEQUENCE || !stores[i].next_number ||
           stores[i].next_number>MAX_CAPTURE_NUMBER+1u) return CAPTURE_INVALID;
        if(stores[i].next_number>maximum) maximum=stores[i].next_number;
    }
    for(i=0;i<count;++i) stores[i].next_number=maximum;
    return CAPTURE_OK;
}

static int same_directory(const char *a, const char *b)
{
    while(*a && *b) {
        if(tolower((unsigned char)*a)!=tolower((unsigned char)*b)) return 0;
        ++a; ++b;
    }
    return *a==*b;
}

int capture_store_save_pair_formats(capture_store *first, capture_store *second,
                            uint32_t width, uint32_t height,
                            unsigned first_format, capture_read_rgb_row first_row, void *first_context,
                            unsigned second_format, capture_read_rgb_row second_row, void *second_context,
                            uint8_t *scratch, size_t capacity,
                            char saved_name[CAPTURE_NAME_CAPACITY])
{
    char name[CAPTURE_NAME_CAPACITY], other[CAPTURE_NAME_CAPACITY], target[CAPTURE_PATH_CAPACITY];
    int result;
    if(saved_name) saved_name[0]='\0';
    result=capture_store_current_name(first,name);
    if(result!=CAPTURE_OK) return result;
    result=capture_store_current_name(second,other);
    if(result!=CAPTURE_OK) return result;
    if(first->mode!=CAPTURE_NUMBERING_SEQUENCE || second->mode!=CAPTURE_NUMBERING_SEQUENCE ||
       first==second || first->fs!=second->fs || first->fs_context!=second->fs_context ||
       same_directory(first->directory,second->directory) || strcmp(name,other)!=0) return CAPTURE_INVALID;
    result=stage_bmp(first,width,height,first_format,first_row,first_context,scratch,capacity);
    if(result!=CAPTURE_OK) return result;
    result=stage_bmp(second,width,height,second_format,second_row,second_context,scratch,capacity);
    if(result!=CAPTURE_OK) { discard_bmp(first); return result; }
    result=commit_bmp(first,name);
    if(result!=CAPTURE_OK) { discard_bmp(first); discard_bmp(second); return result; }
    result=commit_bmp(second,name);
    if(result!=CAPTURE_OK) {
        /* We exclusively created this target in commit1. Existing photographs
         * in either directory are never candidates for rollback/removal. */
        int rollback;
        path_for(first,name,target);
        rollback=first->fs->remove(first->fs_context,target);
        discard_bmp(second);
        if(rollback!=CAPTURE_OK) {
            first->ready=second->ready=0;
            return CAPTURE_PAIR_PARTIAL;
        }
        return result;
    }
    ++first->next_number; ++second->next_number;
    if(saved_name) strcpy(saved_name,name);
    return CAPTURE_OK;
}

int capture_store_save_bmp(capture_store *s, uint32_t w, uint32_t h,
    capture_read_rgb_row row, void *context, uint8_t *scratch, size_t capacity,
    char name[CAPTURE_NAME_CAPACITY])
{
    return capture_store_save_bmp_format(s,w,h,24u,row,context,scratch,capacity,name);
}

int capture_store_save_pair(capture_store *a, capture_store *b, uint32_t w, uint32_t h,
    capture_read_rgb_row row_a, void *context_a, capture_read_rgb_row row_b, void *context_b,
    uint8_t *scratch, size_t capacity, char name[CAPTURE_NAME_CAPACITY])
{
    return capture_store_save_pair_formats(a,b,w,h,24u,row_a,context_a,24u,row_b,context_b,scratch,capacity,name);
}

static int scan_album_entry(void *context, const char *name, int is_directory)
{
    char normalized[10];
    size_t n = strlen(name);
    int bmp = n == 9u && tolower((unsigned char)name[6]) == 'b' &&
              tolower((unsigned char)name[7]) == 'm' && tolower((unsigned char)name[8]) == 'p';
    int json = n == 10u && tolower((unsigned char)name[6]) == 'j' &&
               tolower((unsigned char)name[7]) == 's' && tolower((unsigned char)name[8]) == 'o' &&
               tolower((unsigned char)name[9]) == 'n';
    (void)is_directory; /* A directory can occupy a final filename too. */
    if ((!bmp && !json) || name[5] != '.') return CAPTURE_OK;
    memcpy(normalized, name, 5u);
    memcpy(normalized + 5u, ".bmp", 5u);
    return scan_entry(context, normalized, 0);
}

int capture_store_init_album(capture_store stores[3], const capture_fs_ops *fs,
                            void *context, const char *const directories[3])
{
    scan_state scan;
    unsigned i, song = 0u;
    uint32_t sequence = 0;
    int rc;
    if (!stores || !directories) return CAPTURE_INVALID;
    memset(stores, 0, 3u * sizeof(*stores));
    for (i = 0; i < 3u; ++i) {
        if (!directories[i]) return CAPTURE_INVALID;
    }
    if (same_directory(directories[0],directories[1]) ||
        same_directory(directories[0],directories[2]) ||
        same_directory(directories[1],directories[2])) return CAPTURE_INVALID;
    for (i = 0; i < 100u; ++i) scan.pages[i] = -1;
    scan.newest_song = -1;
    for (i = 0; i < 3u; ++i) {
        rc = capture_store_init(stores+i, fs, context, directories[i]);
        if (rc != CAPTURE_OK) goto failed;
        if (stores[i].song > song) song = stores[i].song;
        if (stores[i].cursor_sequence > sequence) sequence = stores[i].cursor_sequence;
        rc = fs->list(context, stores[i].directory, scan_album_entry, &scan);
        if (rc != CAPTURE_OK) goto failed;
    }
    if (scan.newest_song > (int)song) song = (unsigned)scan.newest_song;
    for (i = 0; i < 3u; ++i) {
        stores[i].song = song;
        stores[i].next_page = (unsigned)(scan.pages[song]+1);
        stores[i].cursor_sequence = sequence;
    }
    return CAPTURE_OK;
failed:
    for (i = 0; i < 3u; ++i) stores[i].ready = 0;
    return rc;
}

static int album_valid(capture_store stores[3])
{
    unsigned i;
    if (!stores) return CAPTURE_INVALID;
    for (i = 0; i < 3u; ++i) {
        if (!stores[i].ready) return CAPTURE_NOT_READY;
        if (stores[i].mode != CAPTURE_NUMBERING_SONG_PAGE ||
            stores[i].fs != stores[0].fs || stores[i].fs_context != stores[0].fs_context ||
            stores[i].song != stores[0].song || stores[i].next_page != stores[0].next_page ||
            stores[i].song > 99u || stores[i].next_page > 100u) return CAPTURE_INVALID;
    }
    if (same_directory(stores[0].directory,stores[1].directory) ||
        same_directory(stores[0].directory,stores[2].directory) ||
        same_directory(stores[1].directory,stores[2].directory)) return CAPTURE_INVALID;
    return CAPTURE_OK;
}

int capture_store_album_next_song(capture_store stores[3])
{
    unsigned i;
    int rc = album_valid(stores);
    if (rc != CAPTURE_OK) return rc;
    rc = capture_store_next_song(stores);
    if (rc != CAPTURE_OK) return rc;
    for (i = 1; i < 3u; ++i) {
        stores[i].song = stores[0].song;
        stores[i].next_page = 0;
        stores[i].cursor_sequence = stores[0].cursor_sequence;
    }
    return CAPTURE_OK;
}

static int stage_metadata(capture_store *s, const uint8_t *data, size_t length)
{
    char tmp[CAPTURE_PATH_CAPACITY];
    void *file = NULL;
    int rc, close_rc;
    path_for(s, "PHOTO.TMP", tmp);
    rc = remove_if_present(s,tmp);
    if (rc != CAPTURE_OK) return rc;
    rc = s->fs->create_exclusive(s->fs_context,tmp,&file);
    if (rc != CAPTURE_OK) return rc;
    rc = write_all(s,file,data,length);
    if (rc == CAPTURE_OK) rc = s->fs->sync(s->fs_context,file);
    close_rc = s->fs->close(s->fs_context,file);
    if (rc == CAPTURE_OK) rc = close_rc;
    if (rc != CAPTURE_OK) (void)remove_if_present(s,tmp);
    return rc;
}

int capture_store_save_album(capture_store stores[3], uint32_t width, uint32_t height,
                            capture_read_rgb_row rgb_row, void *rgb_context,
                            capture_read_rgb_row gray_row, void *gray_context,
                            const uint8_t *metadata, size_t metadata_length,
                            uint8_t *scratch, size_t capacity,
                            char saved_name[CAPTURE_NAME_CAPACITY])
{
    char name[CAPTURE_NAME_CAPACITY], json[CAPTURE_NAME_CAPACITY];
    char from[CAPTURE_PATH_CAPACITY], target[CAPTURE_PATH_CAPACITY];
    unsigned i, committed = 0;
    int rc, rollback_failed = 0;
    if (saved_name) saved_name[0] = '\0';
    rc = album_valid(stores);
    if (rc != CAPTURE_OK) return rc;
    if (!metadata || !metadata_length) return CAPTURE_INVALID;
    rc = capture_store_current_name(stores,name);
    if (rc != CAPTURE_OK) return rc;
    memcpy(json,name,5u); memcpy(json+5u,".json",6u);
    rc = stage_bmp(stores,width,height,CAPTURE_BMP_RGB24,rgb_row,rgb_context,scratch,capacity);
    if (rc != CAPTURE_OK) return rc;
    rc = stage_bmp(stores+1,width,height,CAPTURE_BMP_GRAY8,gray_row,gray_context,scratch,capacity);
    if (rc == CAPTURE_OK) rc = stage_metadata(stores+2,metadata,metadata_length);
    if (rc != CAPTURE_OK) goto cleanup;
    for (i = 0; i < 3u; ++i) {
        path_for(stores+i,"PHOTO.TMP",from);
        path_for(stores+i,i==2u?json:name,target);
        rc = stores[i].fs->rename_no_replace(stores[i].fs_context,from,target);
        if (rc != CAPTURE_OK) break;
        ++committed;
    }
    if (rc == CAPTURE_OK) {
        for (i = 0; i < 3u; ++i) ++stores[i].next_page;
        if (saved_name) strcpy(saved_name,name);
        return CAPTURE_OK;
    }
    for (i = 0; i < committed; ++i) {
        path_for(stores+i,i==2u?json:name,target);
        if (stores[i].fs->remove(stores[i].fs_context,target) != CAPTURE_OK) rollback_failed = 1;
    }
cleanup:
    for (i = 0; i < 3u; ++i) discard_bmp(stores+i);
    if (rollback_failed) {
        for (i = 0; i < 3u; ++i) stores[i].ready = 0;
        return CAPTURE_PAIR_PARTIAL;
    }
    return rc;
}
