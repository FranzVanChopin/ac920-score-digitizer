#ifndef CAPTURE_STORE_H
#define CAPTURE_STORE_H

#include <stddef.h>
#include <stdint.h>

#define CAPTURE_PATH_CAPACITY 128
#define CAPTURE_NAME_CAPACITY 11

enum {
    CAPTURE_OK = 0,
    CAPTURE_FS_IO = -1,
    CAPTURE_FS_NOENT = -2,
    CAPTURE_FS_EXISTS = -3,
    CAPTURE_FS_NO_SPACE = -4,
    CAPTURE_INVALID = -10,
    CAPTURE_PAGE_FULL = -11,
    CAPTURE_SONG_FULL = -12,
    CAPTURE_STATE_CORRUPT = -13,
    CAPTURE_ROW_ERROR = -14,
    CAPTURE_SHORT_WRITE = -15,
    CAPTURE_NOT_READY = -16,
    CAPTURE_NUMBER_FULL = -17,
    CAPTURE_PAIR_PARTIAL = -18
};

enum {
    CAPTURE_NUMBERING_SONG_PAGE = 0,
    CAPTURE_NUMBERING_SEQUENCE = 1
};

typedef int (*capture_list_visitor)(void *visitor_context, const char *name,
                                    int is_directory);

/* All operations return CAPTURE_OK or a negative status. No implicit formatting
 * or media mount is performed. The caller mounts the intended SD volume first.
 * create_exclusive and rename_no_replace MUST reject an existing destination.
 * sync+close must persist file bytes, and rename/remove must persist directory
 * metadata before reporting success (FatFs f_sync/f_close/f_rename/f_unlink).
 * read_small returns the complete length, or an error when capacity is too small.
 * One application owns this directory; callbacks must not reenter this module. */
typedef struct {
    int (*mkdir)(void *context, const char *path);
    int (*list)(void *context, const char *path,
                capture_list_visitor visitor, void *visitor_context);
    int (*read_small)(void *context, const char *path, uint8_t *data,
                      size_t capacity, size_t *length);
    int (*create_exclusive)(void *context, const char *path, void **file);
    int (*write)(void *context, void *file, const uint8_t *data,
                 size_t length, size_t *written);
    int (*sync)(void *context, void *file);
    int (*close)(void *context, void *file);
    int (*rename_no_replace)(void *context, const char *from, const char *to);
    int (*remove)(void *context, const char *path);
    /* Optional: allocate file size, then rewind; only used on our PHOTO.TMP. */
    int (*reserve)(void *context, void *file, uint32_t bytes);
} capture_fs_ops;

typedef struct {
    const capture_fs_ops *fs;
    void *fs_context;
    char directory[CAPTURE_PATH_CAPACITY];
    uint32_t cursor_sequence;
    unsigned song;
    unsigned next_page; /* 100 means current song is full; never rolls silently. */
    unsigned mode;
    unsigned next_number; /* Sequence mode: 1..999999; 1000000 means full. */
    int ready;
} capture_store;

/* Fetch a frozen, complete frame's row as tightly packed R,G,B bytes. y=0 is
 * the top row. The callback must provide width*3 bytes and return CAPTURE_OK.
 * The DMA producer must not overwrite this frame during the save. */
typedef int (*capture_read_rgb_row)(void *context, uint32_t y,
                                    uint8_t *rgb, size_t length);

enum { CAPTURE_BMP_RGB24=24, CAPTURE_BMP_GRAY8=8, CAPTURE_BMP_BINARY1=1 };
/* Indexed formats request width bytes per row: gray codes or binary 0/255.
 * BINARY1 packs MSB first; no threshold or quantization is applied. */
int capture_store_save_bmp_format(capture_store *store, uint32_t width, uint32_t height,
                           unsigned format, capture_read_rgb_row read_row, void *row_context,
                           uint8_t *scratch, size_t scratch_capacity,
                           char saved_name[CAPTURE_NAME_CAPACITY]);
int capture_store_save_pair_formats(capture_store *first, capture_store *second,
                            uint32_t width, uint32_t height,
                            unsigned first_format, capture_read_rgb_row first_row, void *first_context,
                            unsigned second_format, capture_read_rgb_row second_row, void *second_context,
                            uint8_t *scratch, size_t scratch_capacity,
                            char saved_name[CAPTURE_NAME_CAPACITY]);

int capture_store_init(capture_store *store, const capture_fs_ops *fs,
                       void *fs_context, const char *image_directory);
/* Scan six-digit .bmp names (case-insensitive extension), including directory
 * names, and continue at the maximum occupied number + 1, starting from 1.
 * No song cursor files are read or written; read_small may be NULL in this mode.
 * The caller supplies its desired image directory, e.g. "0:/Images". */
int capture_store_init_sequence(capture_store *store, const capture_fs_ops *fs,
                                void *fs_context, const char *image_directory);
int capture_store_current_name(const capture_store *store,
                               char name[CAPTURE_NAME_CAPACITY]);
/* Sequence mode returns CAPTURE_INVALID without filesystem operations. */
int capture_store_next_song(capture_store *store);
/* scratch needs ((width*3+3)&~3) bytes, not another frame buffer. Writes a
 * standards-compliant bottom-up BGR24 BMP. Number advances only after the
 * temporary file has been synced, closed and committed without replacement. */
int capture_store_save_bmp(capture_store *store, uint32_t width, uint32_t height,
                           capture_read_rgb_row read_row, void *row_context,
                           uint8_t *scratch, size_t scratch_capacity,
                           char saved_name[CAPTURE_NAME_CAPACITY]);
/* Advance all sequence cursors to their common maximum. No filesystem writes.
 * Used after scanning Image/Grey/Binary and after a successful capture. */
int capture_store_align_sequences(capture_store *stores, size_t count);
/* Two same-sized images, identical sequence name, different directories on the
 * SAME mounted FS/context. Both temporary files are synced/closed before either
 * is renamed. If commit2 fails, remove ONLY our freshly committed first file.
 * A failed rollback returns PAIR_PARTIAL and makes both stores unready; remount
 * and rescan. FAT cannot promise two-directory atomicity on sudden power loss:
 * rescan+align must skip any surviving orphan number, never overwrite it. */
int capture_store_save_pair(capture_store *first, capture_store *second,
                            uint32_t width, uint32_t height,
                            capture_read_rgb_row first_row, void *first_context,
                            capture_read_rgb_row second_row, void *second_context,
                            uint8_t *scratch, size_t scratch_capacity,
                            char saved_name[CAPTURE_NAME_CAPACITY]);

/* Song/page album: directories are Image, Grey, Meta, in that order. Resume
 * from the greatest occupied SS-PP BMP/JSON name across all three directories,
 * including orphan files and directory collisions. Legacy six-digit files
 * are retained and ignored. Empty-song selection uses the Image cursor journal. */
int capture_store_init_album(capture_store stores[3], const capture_fs_ops *fs,
                            void *context, const char *const directories[3]);
int capture_store_album_next_song(capture_store stores[3]);
/* Freeze one frame for both callbacks. Stage/sync/close both images and JSON
 * before committing; Meta/SS-PP.json is committed LAST as the completion marker.
 * Failed commit rolls back only files created by this call. A rollback failure
 * makes all stores unready; remount/rescan skips every surviving occupied name.
 * FAT cannot guarantee three-directory atomicity across sudden power loss. */
int capture_store_save_album(capture_store stores[3], uint32_t width, uint32_t height,
                            capture_read_rgb_row rgb_row, void *rgb_context,
                            capture_read_rgb_row gray_row, void *gray_context,
                            const uint8_t *metadata, size_t metadata_length,
                            uint8_t *scratch, size_t scratch_capacity,
                            char saved_name[CAPTURE_NAME_CAPACITY]);

#endif
